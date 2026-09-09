"""Export RL prompts or SFT conversations from structured PrimeRL traces.

TraceDataExporter groups repeated rollouts by prompt, computes empirical prompt
pass rates, applies outcome/difficulty/field filters, selects deterministic
quotas, and writes training-ready JSONL.

Difficulty is identical to prompt_pass_rate. Pass rates are computed within
each configured step window. Set window_size=None to use the complete requested
step range as one window.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from itertools import groupby
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence

try:
    import orjson
except ImportError:  # Optional speed-up for large traces.
    orjson = None

from .primerl import PrimeRLAdapter, fingerprint, open_prime_rl_run
from .tracer import RLTracer


class ExportMode(str, Enum):
    """Output row shape."""

    RL = "rl"
    SFT = "sft"


class Acceptance(str, Enum):
    """Which final-label outcomes may become output rows."""

    ANY = "any"
    CORRECT = "correct"
    INCORRECT = "incorrect"


@dataclass(frozen=True)
class TraceRunSpec:
    """One PrimeRL run and its inclusive step range."""

    path: str | Path
    step_start: int = 1
    step_end: int | None = None
    split: str = "train"
    name: str | None = None
    rl_data_path: str | Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if self.rl_data_path is not None:
            object.__setattr__(
                self,
                "rl_data_path",
                Path(self.rl_data_path),
            )
        if self.step_start < 0:
            raise ValueError("step_start must be nonnegative")
        if self.step_end is not None and self.step_end < self.step_start:
            raise ValueError("step_end must not precede step_start")
        if self.split not in {"train", "eval"}:
            raise ValueError("split must be train or eval")


@dataclass(frozen=True)
class DifficultyFilter:
    """Inclusive empirical prompt-difficulty bounds.

    Pass-rate and difficulty bounds may be combined; all supplied bounds must
    match. Difficulty is exactly the pass rate.
    """

    min_pass_rate: float | None = None
    max_pass_rate: float | None = None
    min_difficulty: float | None = None
    max_difficulty: float | None = None

    def __post_init__(self) -> None:
        values = asdict(self)
        for name, value in values.items():
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if (
            self.min_pass_rate is not None
            and self.max_pass_rate is not None
            and self.min_pass_rate > self.max_pass_rate
        ):
            raise ValueError("min_pass_rate must not exceed max_pass_rate")
        if (
            self.min_difficulty is not None
            and self.max_difficulty is not None
            and self.min_difficulty > self.max_difficulty
        ):
            raise ValueError("min_difficulty must not exceed max_difficulty")

    def accepts(self, pass_rate: float | None) -> bool:
        has_bounds = any(value is not None for value in asdict(self).values())
        if pass_rate is None:
            return not has_bounds
        difficulty = pass_rate
        return (
            (self.min_pass_rate is None or pass_rate >= self.min_pass_rate)
            and (self.max_pass_rate is None or pass_rate <= self.max_pass_rate)
            and (self.min_difficulty is None or difficulty >= self.min_difficulty)
            and (self.max_difficulty is None or difficulty <= self.max_difficulty)
        )


@dataclass(frozen=True)
class FieldFilter:
    """Filter a trace using an inclusive dotted record path.

    Examples: metrics.num_turns, rewards.reward.score, and info.env_name.
    """

    path: str
    minimum: float | None = None
    maximum: float | None = None
    allowed_values: tuple[Any, ...] | None = None
    include_missing: bool = False

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("FieldFilter.path must not be empty")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("FieldFilter.minimum must not exceed maximum")
        if self.allowed_values is not None:
            object.__setattr__(self, "allowed_values", tuple(self.allowed_values))

    def accepts(self, record: Mapping[str, Any]) -> bool:
        value = value_at_path(record, self.path)
        if value is None:
            return self.include_missing
        if self.allowed_values is not None and value not in self.allowed_values:
            return False
        if self.minimum is not None:
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or value < self.minimum
            ):
                return False
        if self.maximum is not None:
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or value > self.maximum
            ):
                return False
        return True


@dataclass(frozen=True)
class ExportPreferences:
    """Selection and serialization preferences."""

    mode: ExportMode | str = ExportMode.SFT
    acceptance: Acceptance | str = Acceptance.CORRECT
    difficulty: DifficultyFilter = field(default_factory=DifficultyFilter)
    field_filters: tuple[FieldFilter, ...] = ()
    per_prompt: int = 2
    target_count: int | None = None
    window_size: int | None = None
    unique_prompts: bool = False
    global_per_prompt: int | None = None
    tool_call_sampling_power: float = 0.0
    seed: int = 17
    include_provenance: bool = False
    include_raw_trace: bool = False
    include_tools: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", ExportMode(self.mode))
        object.__setattr__(self, "acceptance", Acceptance(self.acceptance))
        object.__setattr__(self, "field_filters", tuple(self.field_filters))
        if self.per_prompt < 1:
            raise ValueError("per_prompt must be positive")
        if self.target_count is not None and self.target_count < 1:
            raise ValueError("target_count must be positive or None")
        if self.window_size is not None and self.window_size < 1:
            raise ValueError("window_size must be positive or None")
        if self.global_per_prompt is not None and self.global_per_prompt < 1:
            raise ValueError("global_per_prompt must be positive or None")
        if (
            not math.isfinite(self.tool_call_sampling_power)
            or self.tool_call_sampling_power < 0
        ):
            raise ValueError(
                "tool_call_sampling_power must be finite and nonnegative"
            )


@dataclass(frozen=True)
class ExportResult:
    output_path: Path
    summary_path: Path
    selected_count: int
    available_count: int
    eligible_prompt_groups: int


@dataclass(frozen=True)
class _ResolvedRun:
    index: int
    name: str
    path: Path
    step_start: int
    step_end: int
    split: str
    rl_data_path: Path | None


@dataclass(frozen=True)
class _Candidate:
    run_index: int
    run_name: str
    step: int
    window_start: int
    window_end: int
    prompt_key: str
    prompt_preview: str
    trace_path: Path
    line_index: int
    trace_id: str | None
    episode_id: str | None
    data_index: int | None
    tool_call_count: int
    score: str | float
    selection_slot: int = 0
    prompt_pass_rate: float | None = None

    @property
    def group_key(self) -> tuple[int, int, int, str]:
        return self.run_index, self.window_start, self.window_end, self.prompt_key


@dataclass
class _Group:
    trace_count: int = 0
    scored_count: int = 0
    correct_count: int = 0
    candidates: list[_Candidate] = field(default_factory=list)

    @property
    def pass_rate(self) -> float | None:
        return self.correct_count / self.scored_count if self.scored_count else None


def value_at_path(record: Mapping[str, Any], path: str) -> Any:
    """Read a dotted mapping path, returning None when it is absent."""

    value: Any = record
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            return None
        value = value[component]
    return value


def prompt_messages(
    messages: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return context preceding the first sampled assistant turn."""

    result: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "assistant":
            break
        result.append(message)
    if not result:
        raise ValueError(
            "trajectory has no prompt context before its first assistant turn"
        )
    return result


class TraceDataExporter:
    """Select and write training rows from saved structured trace runs."""

    def __init__(
        self,
        runs: Sequence[TraceRunSpec | str | Path],
        preferences: ExportPreferences | None = None,
    ):
        if not runs:
            raise ValueError("at least one run is required")
        self.run_specs = tuple(
            run if isinstance(run, TraceRunSpec) else TraceRunSpec(run)
            for run in runs
        )
        self.preferences = preferences or ExportPreferences()

    def write_jsonl(self, output_path: str | Path) -> ExportResult:
        """Select rows, write JSONL plus a summary, and return export counts."""

        output_path = Path(output_path)
        resolved, open_runs = self._open_runs()
        try:
            groups, window_stats = self._scan(resolved, open_runs)
            eligible = self._eligible_candidates(groups)
            selected = self._choose(eligible)
            summary = self._summary(
                resolved, groups, window_stats, eligible, selected
            )
            self._materialize(
                output_path,
                selected,
                open_runs,
                resolved,
            )
            summary_path = output_path.with_suffix(
                output_path.suffix + ".summary.json"
            )
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return ExportResult(
                output_path,
                summary_path,
                len(selected),
                len(eligible),
                summary["eligible_prompt_groups"],
            )
        finally:
            for run in open_runs.values():
                run.close()

    def inspect(self) -> dict[str, Any]:
        """Run selection without writing data and return the summary."""

        resolved, open_runs = self._open_runs()
        try:
            groups, window_stats = self._scan(resolved, open_runs)
            eligible = self._eligible_candidates(groups)
            selected = self._choose(eligible)
            return self._summary(
                resolved, groups, window_stats, eligible, selected
            )
        finally:
            for run in open_runs.values():
                run.close()

    @staticmethod
    def _resolve_path(path: Path) -> Path:
        path = path.resolve()
        if (path / "rollouts").is_dir():
            return path
        if (path / "run_default" / "rollouts").is_dir():
            return path / "run_default"
        if path.name in {"rollouts", "token_exports"} and path.is_dir():
            return path
        raise ValueError(f"no rollouts directory under {path}")

    def _open_runs(
        self,
    ) -> tuple[list[_ResolvedRun], dict[int, RLTracer]]:
        resolved: list[_ResolvedRun] = []
        open_runs: dict[int, RLTracer] = {}
        try:
            for index, spec in enumerate(self.run_specs):
                path = self._resolve_path(Path(spec.path))
                run = open_prime_rl_run(
                    path,
                    "unused-for-structured-traces",
                    index_path=":memory:",
                )
                if not isinstance(run.adapter, PrimeRLAdapter):
                    raise TypeError(f"expected PrimeRLAdapter for {path}")
                steps = list(run.adapter.list_steps(spec.split))
                if not steps:
                    raise ValueError(
                        f"no {spec.split} structured traces under {path}"
                    )
                step_end = (
                    spec.step_end
                    if spec.step_end is not None
                    else max(steps)
                )
                if step_end < spec.step_start:
                    raise ValueError(
                        f"resolved step range is empty for {path}"
                    )
                experiment = (
                    path.parent if path.name == "run_default" else path
                )
                resolved.append(
                    _ResolvedRun(
                        index,
                        spec.name or experiment.name,
                        path,
                        spec.step_start,
                        step_end,
                        spec.split,
                        (
                            Path(spec.rl_data_path).resolve()
                            if spec.rl_data_path is not None
                            else None
                        ),
                    )
                )
                open_runs[index] = run
            return resolved, open_runs
        except BaseException:
            for run in open_runs.values():
                run.close()
            raise

    def _window(
        self,
        run: _ResolvedRun,
        step: int,
    ) -> tuple[int, int]:
        size = self.preferences.window_size
        if size is None:
            return run.step_start, run.step_end
        start = (
            run.step_start
            + ((step - run.step_start) // size) * size
        )
        return start, min(start + size - 1, run.step_end)

    @staticmethod
    def _outcome(record: Mapping[str, Any]) -> bool | None:
        value = value_at_path(record, "metrics.correct_final_label")
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value >= 1.0
        return None

    def _accepts_record(
        self,
        record: Mapping[str, Any],
        outcome: bool | None,
    ) -> bool:
        acceptance = self.preferences.acceptance
        if (
            acceptance is Acceptance.CORRECT
            and outcome is not True
        ):
            return False
        if (
            acceptance is Acceptance.INCORRECT
            and outcome is not False
        ):
            return False
        return all(
            item.accepts(record)
            for item in self.preferences.field_filters
        )

    def _score(
        self,
        run: _ResolvedRun,
        step: int,
        path: Path,
        line_index: int,
        record: Mapping[str, Any],
        tool_call_count: int,
    ) -> str | float:
        info = record.get("info") or {}
        identity = (
            record.get("id")
            or info.get("episode_id")
            or f"{path}:{line_index}"
        )
        value = (
            f"{self.preferences.seed}:{run.name}:{step}:{identity}"
        )
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        power = self.preferences.tool_call_sampling_power
        if power == 0:
            return digest
        uniform = (int(digest, 16) + 1) / ((1 << 256) + 1)
        return (
            math.log(-math.log(uniform))
            - power * math.log(max(tool_call_count, 1))
        )

    def _scan(
        self,
        resolved: Sequence[_ResolvedRun],
        open_runs: Mapping[int, RLTracer],
    ) -> tuple[
        dict[tuple[int, int, int, str], _Group],
        dict[tuple[int, int, int], Counter[str]],
    ]:
        groups: dict[tuple[int, int, int, str], _Group] = {}
        stats: dict[tuple[int, int, int], Counter[str]] = {}
        quota = (
            1
            if self.preferences.mode is ExportMode.RL
            else self.preferences.per_prompt
        )
        for run_spec in resolved:
            adapter = open_runs[run_spec.index].adapter
            assert isinstance(adapter, PrimeRLAdapter)
            saved_steps = set(adapter.list_steps(run_spec.split))
            for step in range(
                run_spec.step_start,
                run_spec.step_end + 1,
            ):
                if step not in saved_steps:
                    continue
                window_start, window_end = self._window(
                    run_spec, step
                )
                stats_key = (
                    run_spec.index,
                    window_start,
                    window_end,
                )
                window_stats = stats.setdefault(
                    stats_key, Counter()
                )
                for trace_path in adapter._trace_files(step):
                    if run_spec.split not in trace_path.parts:
                        continue
                    for (
                        line_index,
                        messages,
                        record,
                    ) in adapter._iter_trace_file(trace_path):
                        window_stats["trace_count"] += 1
                        prompt_key, preview = fingerprint(
                            messages,
                            adapter.prompt_parser,
                        )
                        key = (
                            run_spec.index,
                            window_start,
                            window_end,
                            prompt_key,
                        )
                        group_state = groups.setdefault(key, _Group())
                        group_state.trace_count += 1
                        outcome = self._outcome(record)
                        if outcome is not None:
                            group_state.scored_count += 1
                            group_state.correct_count += int(outcome)
                            window_stats["scored_count"] += 1
                            window_stats["correct_count"] += int(
                                outcome
                            )
                        if not self._accepts_record(record, outcome):
                            continue
                        info = record.get("info") or {}
                        task_data = (
                            (record.get("task") or {}).get("data")
                            or {}
                        )
                        data_index = task_data.get("idx")
                        if (
                            not isinstance(data_index, int)
                            or isinstance(data_index, bool)
                        ):
                            data_index = None
                        tool_call_count = 0
                        for message in messages:
                            calls = message.get("tool_calls")
                            if isinstance(calls, list):
                                tool_call_count += len(calls)
                            elif calls is not None:
                                tool_call_count += 1
                            elif message.get("tool_call") is not None:
                                tool_call_count += 1
                        candidate = _Candidate(
                            run_spec.index,
                            run_spec.name,
                            step,
                            window_start,
                            window_end,
                            prompt_key,
                            preview,
                            trace_path,
                            line_index,
                            (
                                str(record["id"])
                                if record.get("id") is not None
                                else None
                            ),
                            (
                                str(info["episode_id"])
                                if info.get("episode_id") is not None
                                else None
                            ),
                            data_index,
                            tool_call_count,
                            self._score(
                                run_spec,
                                step,
                                trace_path,
                                line_index,
                                record,
                                tool_call_count,
                            ),
                        )
                        group_state.candidates.append(candidate)
                        group_state.candidates.sort(
                            key=lambda row: row.score
                        )
                        del group_state.candidates[quota:]
        return groups, stats

    def _eligible_candidates(
        self,
        groups: Mapping[
            tuple[int, int, int, str],
            _Group,
        ],
    ) -> list[_Candidate]:
        eligible: list[_Candidate] = []
        for group_state in groups.values():
            if (
                not group_state.candidates
                or not self.preferences.difficulty.accepts(
                    group_state.pass_rate
                )
            ):
                continue
            eligible.extend(
                replace(
                    row,
                    prompt_pass_rate=group_state.pass_rate,
                )
                for row in group_state.candidates
            )
        return eligible

    def _choose(
        self,
        eligible: Sequence[_Candidate],
    ) -> list[_Candidate]:
        by_group: dict[
            tuple[int, int, int, str],
            list[_Candidate],
        ] = {}
        for row in eligible:
            by_group.setdefault(row.group_key, []).append(row)
        maximum_quota = max(
            (len(rows) for rows in by_group.values()),
            default=0,
        )
        ordered: list[_Candidate] = []
        for slot in range(maximum_quota):
            layer = [
                replace(rows[slot], selection_slot=slot + 1)
                for rows in by_group.values()
                if len(rows) > slot
            ]
            if self.preferences.tool_call_sampling_power:
                layer.sort(key=lambda row: row.score)
            else:
                layer.sort(
                    key=lambda row: hashlib.sha256(
                        (
                            f"{self.preferences.seed}:layer:{slot}:"
                            f"{row.run_name}:{row.window_start}:"
                            f"{row.prompt_key}"
                        ).encode()
                    ).hexdigest()
                )
            ordered.extend(layer)
        global_quota = self.preferences.global_per_prompt
        if global_quota is not None:
            by_prompt: dict[str, list[_Candidate]] = {}
            for row in ordered:
                by_prompt.setdefault(row.prompt_key, []).append(row)
            globally_layered: list[_Candidate] = []
            for slot in range(global_quota):
                layer = [
                    rows[slot]
                    for rows in by_prompt.values()
                    if len(rows) > slot
                ]
                if self.preferences.tool_call_sampling_power:
                    layer.sort(key=lambda row: row.score)
                else:
                    layer.sort(
                        key=lambda row: hashlib.sha256(
                            (
                                f"{self.preferences.seed}:global-layer:"
                                f"{slot}:{row.prompt_key}"
                            ).encode()
                        ).hexdigest()
                    )
                globally_layered.extend(layer)
            ordered = globally_layered
        if self.preferences.unique_prompts:
            seen_prompt_keys: set[str] = set()
            unique_rows = []
            for row in ordered:
                if row.prompt_key in seen_prompt_keys:
                    continue
                seen_prompt_keys.add(row.prompt_key)
                unique_rows.append(row)
            ordered = unique_rows
        target = self.preferences.target_count
        if target is not None and target > len(ordered):
            raise ValueError(
                f"requested {target:,} rows, but only "
                f"{len(ordered):,} are available"
            )
        return ordered if target is None else ordered[:target]

    @staticmethod
    def _json_bytes(value: Any) -> bytes:
        if orjson is not None:
            return orjson.dumps(value) + b"\n"
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")

    @staticmethod
    def _open_output(
        path: Path,
        compressed: bool,
    ) -> BinaryIO:
        if compressed:
            return gzip.open(path, "wb", compresslevel=6)
        return path.open("wb")

    @staticmethod
    def _openai_tools(
        tools: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for tool in tools:
            if (
                tool.get("type") == "function"
                and isinstance(tool.get("function"), Mapping)
            ):
                result.append(dict(tool))
            elif tool.get("name"):
                result.append(
                    {
                        "type": "function",
                        "function": {
                            "name": str(tool["name"]),
                            "description": (
                                tool.get("description") or ""
                            ),
                            "parameters": (
                                tool.get("parameters")
                                or {
                                    "type": "object",
                                    "properties": {},
                                }
                            ),
                        },
                    }
                )
        return result

    @staticmethod
    def _sft_messages(
        messages: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        result = copy.deepcopy(list(messages))
        for message_index, message in enumerate(result):
            singular = message.get("tool_call")
            plural = message.get("tool_calls")
            if isinstance(plural, list):
                source_calls = plural
            elif isinstance(singular, dict):
                source_calls = [singular]
            else:
                continue

            nested_calls = []
            for call_index, call in enumerate(source_calls):
                if not isinstance(call, dict):
                    raise ValueError(
                        "tool call must be an object at message "
                        f"{message_index}, call {call_index}"
                    )
                function = call.get("function")
                if isinstance(function, dict):
                    name = function.get("name") or call.get("name")
                    arguments = function.get(
                        "arguments",
                        call.get("arguments", {}),
                    )
                else:
                    name = call.get("name")
                    arguments = call.get("arguments", {})
                if not name:
                    raise ValueError(
                        "tool call has no function name at message "
                        f"{message_index}, call {call_index}"
                    )
                if isinstance(arguments, str):
                    try:
                        json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = json.dumps(
                            arguments,
                            ensure_ascii=False,
                        )
                else:
                    arguments = json.dumps(
                        arguments,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                nested_calls.append(
                    {
                        "id": str(
                            call.get("id")
                            or f"call_{message_index}_{call_index}"
                        ),
                        "type": "function",
                        "function": {
                            "name": str(name),
                            "arguments": arguments,
                        },
                    }
                )
            message.pop("tool_call", None)
            message["tool_calls"] = nested_calls
        return result

    @staticmethod
    def _load_rl_rows(
        runs: Sequence[_ResolvedRun],
    ) -> dict[int, list[dict[str, Any]]]:
        cache: dict[Path, list[dict[str, Any]]] = {}
        result: dict[int, list[dict[str, Any]]] = {}
        for run in runs:
            if run.rl_data_path is None:
                raise ValueError(
                    "RL mode requires TraceRunSpec.rl_data_path "
                    f"for {run.name}"
                )
            path = run.rl_data_path
            if path not in cache:
                if not path.is_file():
                    raise ValueError(
                        f"RL datadev JSONL does not exist: {path}"
                    )
                rows = []
                with path.open(encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, 1):
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise ValueError(
                                f"non-object RL row at "
                                f"{path}:{line_number}"
                            )
                        rows.append(row)
                cache[path] = rows
            result[run.index] = cache[path]
        return result

    def _row(
        self,
        candidate: _Candidate,
        messages: list[dict[str, Any]],
        trace: dict[str, Any],
        rl_row: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if self.preferences.mode is ExportMode.RL:
            if rl_row is None:
                raise RuntimeError(
                    "missing joined datadev RL source row"
                )
            required = (
                "prompt",
                "answer",
                "output",
                "feedback",
                "record_id",
                "rubric_index",
                "source",
                "tools",
            )
            missing = [key for key in required if key not in rl_row]
            if missing:
                raise ValueError(
                    "datadev RL source row is missing fields: "
                    + ", ".join(missing)
                )
            trace_prompt = prompt_messages(messages)
            if trace_prompt != rl_row["prompt"]:
                raise RuntimeError(
                    "trace prompt does not match its joined "
                    f"datadev row at index {candidate.data_index}"
                )
            row: dict[str, Any] = {
                key: copy.deepcopy(rl_row[key])
                for key in required
            }
        else:
            row = {"messages": self._sft_messages(messages)}
            tools = self._openai_tools(trace.get("tools") or [])
            if self.preferences.include_tools and tools:
                row["tools"] = json.dumps(
                    tools,
                    ensure_ascii=False,
                )
        if self.preferences.include_provenance:
            row["trace_provenance"] = {
                "run": candidate.run_name,
                "step": candidate.step,
                "window_start": candidate.window_start,
                "window_end": candidate.window_end,
                "prompt_key": candidate.prompt_key,
                "prompt_preview": candidate.prompt_preview,
                "prompt_pass_rate": candidate.prompt_pass_rate,
                "difficulty": candidate.prompt_pass_rate,
                "tool_call_count": candidate.tool_call_count,
                "selection_slot": candidate.selection_slot,
                "trace_id": candidate.trace_id,
                "episode_id": candidate.episode_id,
                "trace_path": str(candidate.trace_path),
                "trace_line": candidate.line_index + 1,
            }
        if self.preferences.include_raw_trace:
            row["trace"] = trace
        return row

    def _materialize(
        self,
        output_path: Path,
        selected: Sequence[_Candidate],
        open_runs: Mapping[int, RLTracer],
        runs: Sequence[_ResolvedRun],
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rl_rows = (
            self._load_rl_rows(runs)
            if self.preferences.mode is ExportMode.RL
            else {}
        )
        temporary = output_path.with_name(
            output_path.name + ".tmp"
        )
        ordered = sorted(
            selected,
            key=lambda row: (
                row.run_index,
                row.step,
                str(row.trace_path),
                row.line_index,
            ),
        )
        written = 0
        try:
            with self._open_output(
                temporary,
                output_path.suffix == ".gz",
            ) as output:
                for (
                    run_index,
                    trace_path,
                ), rows_iter in groupby(
                    ordered,
                    key=lambda row: (
                        row.run_index,
                        row.trace_path,
                    ),
                ):
                    selected_lines = {
                        row.line_index: row
                        for row in rows_iter
                    }
                    adapter = open_runs[run_index].adapter
                    assert isinstance(adapter, PrimeRLAdapter)
                    for (
                        line_index,
                        messages,
                        trace,
                    ) in adapter._iter_trace_file(trace_path):
                        candidate = selected_lines.get(line_index)
                        if candidate is None:
                            continue
                        joined_row = None
                        if self.preferences.mode is ExportMode.RL:
                            if candidate.data_index is None:
                                raise RuntimeError(
                                    "trace has no task.data.idx: "
                                    f"{candidate.trace_path}:"
                                    f"{candidate.line_index + 1}"
                                )
                            source_rows = rl_rows[run_index]
                            if not (
                                0
                                <= candidate.data_index
                                < len(source_rows)
                            ):
                                raise RuntimeError(
                                    "trace task.data.idx is outside "
                                    "the datadev source: "
                                    f"{candidate.data_index}"
                                )
                            joined_row = source_rows[
                                candidate.data_index
                            ]
                        output.write(
                            self._json_bytes(
                                self._row(
                                    candidate,
                                    messages,
                                    trace,
                                    joined_row,
                                )
                            )
                        )
                        written += 1
            if written != len(selected):
                raise RuntimeError(
                    f"wrote {written:,} of "
                    f"{len(selected):,} selected rows"
                )
            temporary.replace(output_path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def _summary(
        self,
        runs: Sequence[_ResolvedRun],
        groups: Mapping[
            tuple[int, int, int, str],
            _Group,
        ],
        window_stats: Mapping[
            tuple[int, int, int],
            Counter[str],
        ],
        eligible: Sequence[_Candidate],
        selected: Sequence[_Candidate],
    ) -> dict[str, Any]:
        eligible_counts = Counter(
            (
                row.run_index,
                row.window_start,
                row.window_end,
            )
            for row in eligible
        )
        selected_counts = Counter(
            (
                row.run_index,
                row.window_start,
                row.window_end,
            )
            for row in selected
        )

        def tool_call_distribution(
            rows: Sequence[_Candidate],
        ) -> dict[str, Any]:
            counts = Counter(row.tool_call_count for row in rows)
            total = sum(counts.values())
            return {
                "mean": (
                    sum(calls * count for calls, count in counts.items())
                    / total
                    if total
                    else None
                ),
                "counts": {
                    str(calls): count
                    for calls, count in sorted(counts.items())
                },
            }
        eligible_group_count = len(
            {row.group_key for row in eligible}
        )
        windows = []
        run_by_index = {run.index: run for run in runs}
        for key in sorted(window_stats):
            run_index, start, end = key
            stats = window_stats[key]
            windows.append(
                {
                    "run": run_by_index[run_index].name,
                    "step_start": start,
                    "step_end": end,
                    "trace_count": stats["trace_count"],
                    "scored_count": stats["scored_count"],
                    "correct_count": stats["correct_count"],
                    "eligible_count": eligible_counts[key],
                    "selected_count": selected_counts[key],
                }
            )
        return {
            "preferences": {
                "mode": self.preferences.mode.value,
                "acceptance": self.preferences.acceptance.value,
                "difficulty": asdict(
                    self.preferences.difficulty
                ),
                "field_filters": [
                    asdict(item)
                    for item in self.preferences.field_filters
                ],
                "per_prompt": self.preferences.per_prompt,
                "effective_per_prompt": (
                    1
                    if self.preferences.mode is ExportMode.RL
                    else self.preferences.per_prompt
                ),
                "target_count": self.preferences.target_count,
                "window_size": self.preferences.window_size,
                "unique_prompts": self.preferences.unique_prompts,
                "global_per_prompt": self.preferences.global_per_prompt,
                "tool_call_sampling_power": (
                    self.preferences.tool_call_sampling_power
                ),
                "seed": self.preferences.seed,
                "include_provenance": (
                    self.preferences.include_provenance
                ),
                "include_raw_trace": (
                    self.preferences.include_raw_trace
                ),
                "include_tools": self.preferences.include_tools,
            },
            "runs": [
                {
                    "name": run.name,
                    "path": str(run.path),
                    "step_start": run.step_start,
                    "step_end": run.step_end,
                    "split": run.split,
                    "rl_data_path": (
                        str(run.rl_data_path)
                        if run.rl_data_path is not None
                        else None
                    ),
                }
                for run in runs
            ],
            "available_count": len(eligible),
            "selected_count": len(selected),
            "eligible_prompt_groups": eligible_group_count,
            "eligible_unique_prompts": len(
                {row.prompt_key for row in eligible}
            ),
            "observed_prompt_groups": len(groups),
            "available_tool_call_distribution": (
                tool_call_distribution(eligible)
            ),
            "selected_tool_call_distribution": (
                tool_call_distribution(selected)
            ),
            "windows": windows,
        }


__all__ = [
    "Acceptance",
    "DifficultyFilter",
    "ExportMode",
    "ExportPreferences",
    "ExportResult",
    "FieldFilter",
    "TraceDataExporter",
    "TraceRunSpec",
    "prompt_messages",
    "value_at_path",
]
