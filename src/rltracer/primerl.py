from __future__ import annotations
import hashlib
import json
import re
import sqlite3

try:
    import orjson
except ImportError:  # Optional speed-up for multi-gigabyte trace exports.
    orjson = None

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Callable

from .model import Trajectory
from .tracer import RLTracer

TURN = re.compile(
    r"<\|im_start\|>(?P<role>system|user|assistant|tool)\n"
    r"(?P<content>.*?)(?=<\|im_end\|>|\Z)",
    re.DOTALL,
)


TOOL_CALL = re.compile(
    r"<tool_call>\s*<function=(?P<name>[^>]+)>(?P<body>.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
PARAMETER = re.compile(r"<parameter=(?P<name>[^>]+)>\s*(?P<value>.*?)\s*</parameter>", re.DOTALL)


def parse_qwen_messages(text: str) -> list[dict[str, object]]:
    """Parse native Qwen roles and recover tool calls/responses best-effort."""
    messages: list[dict[str, object]] = []
    pending_tool_name: str | None = None
    for match in TURN.finditer(text):
        role, content = match["role"], match["content"]
        if role == "assistant":
            calls = []
            for tool_match in TOOL_CALL.finditer(content):
                arguments = {item["name"]: item["value"].strip() for item in PARAMETER.finditer(tool_match["body"])}
                pending_tool_name = tool_match["name"].strip()
                calls.append({"type": "function", "function": {"name": pending_tool_name, "arguments": arguments}})
            visible_content = TOOL_CALL.sub("", content).strip() if calls else content
            item: dict[str, object] = {"role": role, "content": visible_content}
            if calls:
                item["tool_calls"] = calls
            messages.append(item)
            continue
        if role == "user" and content.lstrip().startswith("<tool_response>"):
            tool_content = content.lstrip()[len("<tool_response>"):].strip()
            if tool_content.endswith("</tool_response>"):
                tool_content = tool_content[: -len("</tool_response>")].rstrip()
            item = {"role": "tool", "content": tool_content}
            if pending_tool_name:
                item["name"] = pending_tool_name
            messages.append(item)
            pending_tool_name = None
            continue
        messages.append({"role": role, "content": content})
    return messages


def split_packed_conversations(messages: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    result: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    for message in messages:
        if message["role"] == "system" and current:
            result.append(current)
            current = []
        current.append(message)
    return result + ([current] if current else [])


CRITERIA = re.compile(r"<(?:criteria|evaluation_criteria)>\s*(?P<value>.*?)\s*</(?:criteria|evaluation_criteria)>", re.DOTALL | re.IGNORECASE)


def default_prompt_parser(content: str) -> str:
    """Produce a compact display label from the first user message."""
    match = CRITERIA.search(content)
    selected = match["value"] if match else content
    return " ".join(selected.split())[:240]


def fingerprint(messages: list[dict[str, object]], parser: Callable[[str], str]) -> tuple[str, str]:
    """Group trajectories by their first user prompt, excluding system text."""
    first_user = next((str(message["content"]) for message in messages if message["role"] == "user"), "")
    key = hashlib.sha256(first_user.encode("utf-8")).hexdigest()[:20]
    return key, parser(first_user)


class PrimeRLAdapter:
    def __init__(
        self,
        rollout_dir: str | Path,
        tokenizer_name: str,
        index_path: str | Path | None = None,
        prompt_parser: Callable[[str], str] | None = None,
    ):
        # PrimeRL 0.7.1 writes JSONL token exports under token_exports/step_*/rank_*.jsonl.
        # Keep the historical attribute name and support legacy exports.
        self.rollout_dir = Path(rollout_dir)
        self.tokenizer_name = tokenizer_name
        self.prompt_parser = prompt_parser or default_prompt_parser
        self.db = sqlite3.connect(index_path or self.rollout_dir / ".rltracer.sqlite")
        self.db.row_factory = sqlite3.Row
        self.db.executescript(
            "CREATE TABLE IF NOT EXISTS indexed_steps(step INTEGER PRIMARY KEY, signature TEXT);"
            "CREATE TABLE IF NOT EXISTS trajectories(id INTEGER PRIMARY KEY, step INTEGER, prompt_key TEXT, preview TEXT, shard TEXT, group_i INTEGER, row_i INTEGER, conversation_i INTEGER, token_count INTEGER);"
            "CREATE INDEX IF NOT EXISTS by_step_prompt ON trajectories(step, prompt_key, id);"
        )
        self.tokenizer: Any | None = None

    def _tok(self) -> Any:
        if self.tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as error:
                raise RuntimeError(
                    "PrimeRL tokenizer loading requires rltracer[primerl]"
                ) from error
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, trust_remote_code=True)
        return self.tokenizer

    def list_steps(self, split: str | None = None) -> Sequence[int]:
        steps = sorted(int(path.name[5:]) for path in self.rollout_dir.glob("step_*") if path.name[5:].isdigit())
        if split is None:
            return steps
        return [
            step for step in steps
            if any((self.rollout_dir / f"step_{step}" / split).glob("**/traces.jsonl"))
        ]

    def _shards(self, step: int) -> list[Path]:
        step_dir = self.rollout_dir / f"step_{step}"
        jsonl = sorted(step_dir.glob("rank_*.jsonl"))
        if jsonl:
            return jsonl
        return sorted(step_dir.glob("rank_*.bin"))

    def _trace_files(self, step: int) -> list[Path]:
        step_dir = self.rollout_dir / f"step_{step}"
        all_traces = sorted(step_dir.glob("**/all/traces.jsonl"))
        return all_traces or sorted(step_dir.glob("**/traces.jsonl"))

    def _iter(self, path: Path) -> Iterator[tuple[int, int, list[int]]]:
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for row_index, line in enumerate(handle):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        # Ignore a final partial line from an active export.
                        continue
                    token_ids = record.get("token_ids") if isinstance(record, dict) else None
                    if isinstance(token_ids, list) and all(
                        isinstance(token_id, int) and not isinstance(token_id, bool)
                        for token_id in token_ids
                    ):
                        yield 0, row_index, token_ids
            return

        import msgpack
        with path.open("rb") as handle:
            for payload in msgpack.Unpacker(handle, raw=False):
                for group_index, group in enumerate(payload):
                    if not isinstance(group, list):
                        continue
                    for row_index, token_ids in enumerate(group):
                        # Prime-RL stores token sequences alongside float-valued
                        # logprobs, rewards, masks, and optional None fields.
                        if isinstance(token_ids, list) and all(
                            isinstance(token_id, int) and not isinstance(token_id, bool)
                            for token_id in token_ids
                        ):
                            yield group_index, row_index, token_ids

    @staticmethod
    def _trace_messages(record: dict[str, Any]) -> list[dict[str, object]]:
        messages: list[dict[str, object]] = []
        for node in record.get("nodes", []):
            if not isinstance(node, dict):
                continue
            message = node.get("message")
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if not isinstance(role, str):
                continue
            item = {key: value for key, value in message.items() if key != "role"}
            item["role"] = role
            item["content"] = str(item.get("content") or "")
            messages.append(item)
        return messages

    def _iter_trace_file(self, path: Path) -> Iterator[tuple[int, list[dict[str, object]], dict[str, Any]]]:
        with path.open(encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                try:
                    record = orjson.loads(line) if orjson else json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(record, dict):
                    continue
                messages = self._trace_messages(record)
                if messages:
                    yield line_index, messages, record

    def ensure_step_indexed(self, step: int) -> None:
        shards = self._shards(step)
        trace_files = self._trace_files(step)
        sources = trace_files or shards
        signature = "prime-rl-traces-v4-first-user-prompt:" + hashlib.sha256("|".join(f"{p}:{p.stat().st_mtime_ns}:{p.stat().st_size}" for p in sources).encode()).hexdigest()
        existing = self.db.execute("SELECT signature FROM indexed_steps WHERE step=?", (step,)).fetchone()
        if existing and existing["signature"] == signature:
            return
        self.db.execute("DELETE FROM trajectories WHERE step=?", (step,))
        if trace_files:
            for trace_file in trace_files:
                for line_index, messages, _record in self._iter_trace_file(trace_file):
                    key, preview = fingerprint(messages, self.prompt_parser)
                    self.db.execute(
                        "INSERT INTO trajectories(step,prompt_key,preview,shard,group_i,row_i,conversation_i,token_count) VALUES(?,?,?,?,?,?,?,?)",
                        (step, key, preview, str(trace_file), -1, line_index, 0, 0),
                    )
        else:
            for shard in shards:
                for group_index, row_index, token_ids in self._iter(shard):
                    decoded = self._tok().decode(token_ids, skip_special_tokens=False)
                    for conversation_index, messages in enumerate(split_packed_conversations(parse_qwen_messages(decoded))):
                        key, preview = fingerprint(messages, self.prompt_parser)
                        self.db.execute(
                            "INSERT INTO trajectories(step,prompt_key,preview,shard,group_i,row_i,conversation_i,token_count) VALUES(?,?,?,?,?,?,?,?)",
                            (step, key, preview, str(shard), group_index, row_index, conversation_index, len(token_ids)),
                        )
        self.db.execute("INSERT OR REPLACE INTO indexed_steps VALUES(?,?)", (step, signature))
        self.db.commit()

    def list_prompts(self, step: int, split: str | None = None) -> Sequence[tuple[str, int, str]]:
        where, params = "step=?", [step]
        if split:
            where += " AND shard LIKE ?"
            params.append(f"%/{split}/%")
        rows = self.db.execute(
            f"SELECT prompt_key,COUNT(*) n,MIN(preview) preview FROM trajectories WHERE {where} GROUP BY prompt_key ORDER BY prompt_key",
            params,
        )
        return [(row["prompt_key"], row["n"], row["preview"]) for row in rows]

    def list_trajectory_ids(self, step: int, prompt_key: str, split: str | None = None) -> Sequence[int]:
        where, params = "step=? AND prompt_key=?", [step, prompt_key]
        if split:
            where += " AND shard LIKE ?"
            params.append(f"%/{split}/%")
        return [row["id"] for row in self.db.execute(f"SELECT id FROM trajectories WHERE {where} ORDER BY id", params)]

    def trajectory_split(self, trajectory_id: int) -> str | None:
        """Return the rollout split encoded in a trace path, when present."""
        row = self.db.execute("SELECT shard FROM trajectories WHERE id=?", (trajectory_id,)).fetchone()
        if row is None:
            raise KeyError(trajectory_id)
        for part in reversed(Path(row["shard"]).parts):
            if part in {"train", "eval"}:
                return part
        return None

    def load_trajectory(self, trajectory_id: int) -> Trajectory:
        row = self.db.execute("SELECT * FROM trajectories WHERE id=?", (trajectory_id,)).fetchone()
        if row is None:
            raise KeyError(trajectory_id)
        shard = Path(row["shard"])
        if shard.name == "traces.jsonl":
            for line_index, messages, record in self._iter_trace_file(shard):
                if line_index == row["row_i"]:
                    metadata = dict(row)
                    metadata["split"] = self.trajectory_split(trajectory_id)
                    metadata["trace"] = record
                    return Trajectory(trajectory_id, row["step"], row["prompt_key"], messages, metadata)
            raise KeyError(f"Trace line {row['row_i']} no longer exists in {shard}")
        token_ids = next(
            tokens for group_index, row_index, tokens in self._iter(shard)
            if group_index == row["group_i"] and row_index == row["row_i"]
        )
        conversations = split_packed_conversations(parse_qwen_messages(self._tok().decode(token_ids, skip_special_tokens=False)))
        metadata = dict(row)
        metadata["split"] = self.trajectory_split(trajectory_id)
        return Trajectory(trajectory_id, row["step"], row["prompt_key"], conversations[row["conversation_i"]], metadata)

    def close(self) -> None:
        self.db.close()


def open_prime_rl_run(
    root: str | Path,
    tokenizer_name: str,
    index_path: str | Path | None = None,
) -> RLTracer:
    root = Path(root)
    if root.name in {"rollouts", "token_exports"}:
        rollouts = root
    elif any((root / "rollouts").glob("step_*/**/traces.jsonl")):
        # A complete run often contains both token exports and structured
        # rollout traces. Prefer traces: they carry split and reward metadata.
        rollouts = root / "rollouts"
    elif (root / "token_exports").is_dir():
        rollouts = root / "token_exports"
    else:
        rollouts = root / "rollouts"
    return RLTracer(PrimeRLAdapter(rollouts, tokenizer_name, index_path))
