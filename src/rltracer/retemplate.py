"""Retemplate RLTracer SFT exports from their selected Datadev JRows.

The saved RL trace contains a rendered system/user prefix, but not the
individual input and rubric fields used to render it. Datadev's selected-JRow
parquet preserves those fields. This module joins each exported trace to its
JRow by rendering the original template, then renders the replacement prefix
without parsing prose out of the old prompt.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class MixTemplateSource:
    """One selected-JRow parquet and the template used for its RL prompts."""

    parquet_path: str | Path
    prompt_template_path: str | Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "parquet_path", Path(self.parquet_path))
        object.__setattr__(
            self,
            "prompt_template_path",
            Path(self.prompt_template_path),
        )


def _load_template(path: Path) -> dict[str, Any]:
    template = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(template, dict):
        raise ValueError(f"template is not an object: {path}")
    return template


def _rubric_text(rubric: Mapping[str, Any]) -> str:
    scores = rubric.get("scores") or {}
    if not isinstance(scores, Mapping):
        raise ValueError("JRow rubric scores are not an object")
    # Selected JRows are serialized with ``sort_keys=True``, which changes a
    # binary rubric from the materializer's display order (pass, fail) to
    # (fail, pass). Prime-RL prompts were rendered before that serialization.
    # Restore the binary display order so hashes describe the actual RL task.
    items = list(scores.items())
    if {str(key).strip().casefold() for key, _ in items} == {"pass", "fail"}:
        order = {"pass": 0, "fail": 1}
        items.sort(key=lambda item: order[str(item[0]).strip().casefold()])
    return "\n".join(
        f"{key} - {value}"
        for key, value in items
        if str(key).strip().casefold() != "weight"
    )


def _render_prefix(
    template: Mapping[str, Any],
    jrow: Mapping[str, Any],
) -> list[dict[str, str]]:
    rubric = jrow.get("rubric") or {}
    if not isinstance(rubric, Mapping):
        raise ValueError("JRow rubric is not an object")
    values = {
        "inputs": jrow.get("input") or "",
        "output": jrow.get("output") or "",
        "evaluation_criteria": rubric.get("criteria") or "",
        "rubric": _rubric_text(rubric),
    }
    missing = set(template.get("input_variables") or ()) - set(values)
    if missing:
        raise ValueError(f"unsupported template variables: {sorted(missing)}")
    rendered = [
        {
            "role": str(message["role"]),
            "content": str(message.get("content") or "").format(**values),
        }
        for message in template.get("chat_template") or ()
        if isinstance(message, Mapping)
        and message.get("role") in {"system", "user"}
    ]
    if len(rendered) < 2:
        raise ValueError("template must render a system/user prefix")
    return rendered


def _prompt_prefix(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, str]], int]:
    prefix: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if message.get("role") == "assistant":
            if not prefix:
                raise ValueError("SFT row has no prompt before its first assistant turn")
            return prefix, index
        prefix.append(
            {
                "role": str(message.get("role") or ""),
                "content": str(message.get("content") or ""),
            }
        )
    raise ValueError("SFT row has no assistant turn")


def _prefix_key(messages: Sequence[Mapping[str, Any]]) -> str:
    normalized = [
        {
            "role": str(message.get("role") or ""),
            "content": str(message.get("content") or ""),
        }
        for message in messages
    ]
    return hashlib.sha256(
        json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _openai_tools(tools: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") == "function" and isinstance(
            tool.get("function"), Mapping
        ):
            result.append(copy.deepcopy(dict(tool)))
        elif tool.get("name"):
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": str(tool["name"]),
                        "description": str(tool.get("description") or ""),
                        "parameters": copy.deepcopy(
                            tool.get("parameters")
                            or {"type": "object", "properties": {}}
                        ),
                    },
                }
            )
    return result


class SFTTemplateRewriter:
    """Replace SFT prompt prefixes using fields recovered from mix parquets."""

    def __init__(
        self,
        sources: Sequence[MixTemplateSource],
        template_path: str | Path,
    ) -> None:
        if not sources:
            raise ValueError("at least one mix-template source is required")
        self.sources = tuple(sources)
        self.template_path = Path(template_path)
        self.template = _load_template(self.template_path)
        self.tools = _openai_tools(self.template.get("tools") or ())
        self._index: sqlite3.Connection | None = None
        self._index_path: Path | None = None
        self._prefix_count = 0

    def _build_index(self) -> None:
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError(
                "reading selected-JRow parquets requires pyarrow"
            ) from error

        descriptor, index_name = tempfile.mkstemp(
            prefix="rltracer-retemplate-", suffix=".sqlite3"
        )
        os.close(descriptor)
        index_path = Path(index_name)
        connection = sqlite3.connect(index_path)
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute(
                "CREATE TABLE prefixes ("
                "prefix_key TEXT PRIMARY KEY, prefix_json TEXT NOT NULL)"
            )
            for source in self.sources:
                parquet_path = Path(source.parquet_path).resolve()
                if not parquet_path.is_file():
                    raise ValueError(
                        f"mix parquet does not exist: {parquet_path}"
                    )
                source_template = _load_template(
                    Path(source.prompt_template_path).resolve()
                )
                parquet = pq.ParquetFile(parquet_path)
                if "jrow_json" not in parquet.schema_arrow.names:
                    raise ValueError(
                        f"mix parquet has no jrow_json column: {parquet_path}"
                    )
                # JRows can contain very large evaluated outputs and web
                # metadata. Keep both parquet decoding and the prompt lookup
                # bounded: replacements live in this temporary disk index.
                for batch in parquet.iter_batches(
                    batch_size=32,
                    columns=["jrow_json"],
                ):
                    for raw in batch.column(0).to_pylist():
                        jrow = json.loads(raw)
                        old_prefix = _render_prefix(source_template, jrow)
                        new_prefix = _render_prefix(self.template, jrow)
                        key = _prefix_key(old_prefix)
                        prefix_json = json.dumps(
                            new_prefix,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        existing = connection.execute(
                            "SELECT prefix_json FROM prefixes "
                            "WHERE prefix_key = ?",
                            (key,),
                        ).fetchone()
                        if existing is not None and existing[0] != prefix_json:
                            raise ValueError(
                                "ambiguous mix-parquet prompt renders to "
                                f"multiple replacement prefixes: {key}"
                            )
                        if existing is None:
                            connection.execute(
                                "INSERT INTO prefixes VALUES (?, ?)",
                                (key, prefix_json),
                            )
                    connection.commit()
            count = connection.execute(
                "SELECT COUNT(*) FROM prefixes"
            ).fetchone()[0]
            if not count:
                raise ValueError("mix parquets contained no JRows")
        except BaseException:
            connection.close()
            index_path.unlink(missing_ok=True)
            raise
        self._index = connection
        self._index_path = index_path
        self._prefix_count = int(count)

    def _replacement_prefix(
        self, key: str
    ) -> list[dict[str, str]] | None:
        if self._index is None:
            self._build_index()
        assert self._index is not None
        match = self._index.execute(
            "SELECT prefix_json FROM prefixes WHERE prefix_key = ?", (key,)
        ).fetchone()
        return None if match is None else json.loads(match[0])

    def close(self) -> None:
        if self._index is not None:
            self._index.close()
            self._index = None
        if self._index_path is not None:
            self._index_path.unlink(missing_ok=True)
            self._index_path = None

    def __enter__(self) -> "SFTTemplateRewriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def rewrite_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise ValueError("SFT row has no messages list")
        old_prefix, first_assistant = _prompt_prefix(messages)
        key = _prefix_key(old_prefix)
        new_prefix = self._replacement_prefix(key)
        if new_prefix is None:
            raise ValueError(f"no selected-JRow match for prompt prefix {key}")
        rewritten = copy.deepcopy(dict(row))
        rewritten["messages"] = [
            *copy.deepcopy(new_prefix),
            *copy.deepcopy(messages[first_assistant:]),
        ]
        if self.tools:
            rewritten["tools"] = json.dumps(self.tools, ensure_ascii=False)
        else:
            rewritten.pop("tools", None)
        return rewritten

    def rewrite_jsonl(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> dict[str, Any]:
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(output_path.name + ".tmp")
        count = 0
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
                    destination.write(
                        json.dumps(
                            self.rewrite_row(row),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    count += 1
            temporary.replace(output_path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return {
            "input_path": str(input_path.resolve()),
            "output_path": str(output_path.resolve()),
            "row_count": count,
            "template_path": str(self.template_path.resolve()),
            "template_sha256": hashlib.sha256(
                self.template_path.read_bytes()
            ).hexdigest(),
            "mix_parquets": [
                str(Path(source.parquet_path).resolve())
                for source in self.sources
            ],
            "matched_prefixes": self._prefix_count,
        }


def main(
    input: Sequence[str],
    output: Sequence[str],
    mix_parquet: Sequence[str],
    source_template: str,
    template: str,
    summary: str | None = None,
) -> None:
    """Rewrite SFT prefixes; list arguments use Fire's Python-list syntax."""
    if len(input) != len(output):
        raise ValueError("provide one output for every input")
    with SFTTemplateRewriter(
        [MixTemplateSource(path, source_template) for path in mix_parquet],
        template,
    ) as rewriter:
        results = [
            rewriter.rewrite_jsonl(input_path, output_path)
            for input_path, output_path in zip(
                input, output, strict=True
            )
        ]
    report = {"rewrites": results}
    if summary:
        summary_path = Path(summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def cli() -> None:
    import fire

    fire.Fire(main)


if __name__ == "__main__":
    cli()
