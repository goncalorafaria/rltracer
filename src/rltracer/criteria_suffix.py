"""Add deterministic criteria suffixes to rows from selected RL runs.

RLTracer exports can omit provenance to retain the exact Prime-RL SFT schema.
This module uses an aligned provenance-enabled replay as a sidecar: it verifies
the assistant/tool suffix for every selected row, builds a content-addressed
run lookup, and writes clean SFT rows with criteria changes only where the run
name matches an explicitly supplied selector.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


OPEN_TAG = "<evaluation_criteria>\n"
CLOSE_TAG = "\n</evaluation_criteria>"


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _first_assistant(messages: Sequence[Mapping[str, Any]]) -> int:
    for index, message in enumerate(messages):
        if message.get("role") == "assistant":
            return index
    raise ValueError("SFT row has no assistant turn")


def _trace_suffix(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("SFT row has no messages list")
    return messages[_first_assistant(messages) :]


def _prompt_hash(row: Mapping[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("SFT row has no messages list")
    return _canonical_hash(messages[: _first_assistant(messages)])


class SearchCriteriaSuffixAugmenter:
    """Append one deterministic suffix to criteria from matching run names."""

    def __init__(
        self,
        aligned_sft_path: str | Path,
        provenance_replay_path: str | Path,
        suffixes: Sequence[str],
        *,
        run_name_substrings: Sequence[str],
    ) -> None:
        if not suffixes or any(not item.strip() for item in suffixes):
            raise ValueError("criteria suffixes must be non-empty strings")
        if not run_name_substrings:
            raise ValueError("at least one run-name substring is required")
        self.aligned_sft_path = Path(aligned_sft_path)
        self.provenance_replay_path = Path(provenance_replay_path)
        self.suffixes = tuple(item.strip() for item in suffixes)
        self.run_name_substrings = tuple(run_name_substrings)
        self._run_by_row_hash: dict[str, bool] | None = None
        self._aligned_counts: Counter[str] = Counter()

    def _is_selected_run(self, run_name: str) -> bool:
        return any(token in run_name for token in self.run_name_substrings)

    def _build_run_lookup(self) -> dict[str, bool]:
        lookup: dict[str, bool] = {}
        counts: Counter[str] = Counter()
        with self.aligned_sft_path.open(encoding="utf-8") as clean_file, \
            self.provenance_replay_path.open(encoding="utf-8") as replay_file:
            for line_number, (clean_line, replay_line) in enumerate(
                zip(clean_file, replay_file, strict=True), 1
            ):
                clean = json.loads(clean_line)
                replay = json.loads(replay_line)
                provenance = replay.get("trace_provenance")
                if not isinstance(provenance, Mapping):
                    raise ValueError(
                        f"missing trace_provenance at replay row {line_number}"
                    )
                if _trace_suffix(clean) != _trace_suffix(replay):
                    raise ValueError(
                        "provenance replay is not aligned with the SFT export "
                        f"at row {line_number}"
                    )
                run_name = str(provenance.get("run") or "")
                if not run_name:
                    raise ValueError(
                        f"empty provenance run at replay row {line_number}"
                    )
                selected = self._is_selected_run(run_name)
                key = _canonical_hash(clean)
                existing = lookup.get(key)
                if existing is not None and existing != selected:
                    raise ValueError(
                        "identical SFT rows have conflicting run provenance: "
                        f"{key}"
                    )
                lookup[key] = selected
                counts["rows"] += 1
                counts["selected_run_rows" if selected else "other_run_rows"] += 1
        if not lookup:
            raise ValueError("aligned SFT export is empty")
        if not counts["selected_run_rows"]:
            raise ValueError("no provenance run matched the requested selectors")
        self._aligned_counts = counts
        return lookup

    @property
    def run_by_row_hash(self) -> dict[str, bool]:
        if self._run_by_row_hash is None:
            self._run_by_row_hash = self._build_run_lookup()
        return self._run_by_row_hash

    def _suffix_index(self, row: Mapping[str, Any]) -> int:
        return int(_prompt_hash(row)[:16], 16) % len(self.suffixes)

    def augment_row(
        self, row: Mapping[str, Any]
    ) -> tuple[dict[str, Any], int | None]:
        key = _canonical_hash(row)
        selected = self.run_by_row_hash.get(key)
        if selected is None:
            raise ValueError(f"SFT row is absent from aligned provenance: {key}")
        result = copy.deepcopy(dict(row))
        if not selected:
            return result, None
        messages = result.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError("SFT row has no system/user prompt")
        user = messages[1]
        if user.get("role") != "user" or not isinstance(user.get("content"), str):
            raise ValueError("SFT row's second message is not a textual user prompt")
        content = user["content"]
        if content.count(OPEN_TAG) != 1 or content.count(CLOSE_TAG) != 1:
            raise ValueError("SFT prompt has an unexpected evaluation_criteria shape")
        index = self._suffix_index(row)
        suffix = self.suffixes[index]
        # Some search datasets already contain a dedicated cross-cutting
        # source-verification rubric drawn from this same phrase set. Do not
        # duplicate that instruction inside its own criterion.
        if any(candidate in content for candidate in self.suffixes):
            return result, -1
        user["content"] = content.replace(
            CLOSE_TAG,
            f"\n\n{suffix}{CLOSE_TAG}",
            1,
        )
        return result, index

    def augment_jsonl(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> dict[str, Any]:
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(output_path.name + ".tmp")
        variants: Counter[int] = Counter()
        rows = 0
        changed = 0
        already_source_checking = 0
        other_run_rows = 0
        try:
            with input_path.open(encoding="utf-8") as source, temporary.open(
                "w", encoding="utf-8"
            ) as destination:
                for line_number, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, Mapping):
                        raise ValueError(
                            f"non-object SFT row at {input_path}:{line_number}"
                        )
                    updated, index = self.augment_row(row)
                    destination.write(
                        json.dumps(
                            updated,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    rows += 1
                    if index == -1:
                        already_source_checking += 1
                    elif index is None:
                        other_run_rows += 1
                    else:
                        changed += 1
                        variants[index] += 1
            temporary.replace(output_path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return {
            "input_path": str(input_path.resolve()),
            "output_path": str(output_path.resolve()),
            "row_count": rows,
            "criteria_augmented_rows": changed,
            "criteria_already_source_checking_rows": already_source_checking,
            "other_run_rows": other_run_rows,
            "unchanged_rows": already_source_checking + other_run_rows,
            "suffix_variant_count": len(self.suffixes),
            "suffix_variant_rows": {
                str(index): variants[index] for index in sorted(variants)
            },
            "run_name_substrings": list(self.run_name_substrings),
            "aligned_sft_path": str(self.aligned_sft_path.resolve()),
            "provenance_replay_path": str(
                self.provenance_replay_path.resolve()
            ),
            "aligned_counts": dict(self._aligned_counts),
        }
