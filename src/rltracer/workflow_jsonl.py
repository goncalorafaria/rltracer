"""Direct adapter for intermediate JSONL emitted by JTC verifier workflows."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .model import Trajectory
from .primerl import default_prompt_parser, fingerprint


def _decoded(value: Any, fallback: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return fallback
    return value if value is not None else fallback


def _messages(record: dict[str, Any]) -> list[dict[str, object]]:
    row = record.get("final_row", record)
    if not isinstance(row, dict):
        return []
    rubric = row.get("rubric")
    if not isinstance(rubric, dict):
        return []
    source = _decoded(rubric.get("messages"), [])
    if not isinstance(source, list):
        return []
    result: list[dict[str, object]] = []
    for message in source:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            continue
        item = {key: value for key, value in message.items() if key != "role"}
        item["role"] = message["role"]
        item["content"] = str(item.get("content") or "")
        result.append(item)
    return result


class WorkflowJSONLAdapter:
    """Lazy reader for one verifier workflow's intermediate prediction JSONL."""

    def __init__(self, jsonl_path: str | Path, index_path: str | Path | None = None):
        self.jsonl_path = Path(jsonl_path).resolve()
        if not self.jsonl_path.is_file():
            raise ValueError(f"Workflow JSONL file does not exist: {self.jsonl_path}")
        self.rollout_dir = self.jsonl_path.parent
        self.prompt_parser = default_prompt_parser
        self.db = sqlite3.connect(index_path or self.jsonl_path.with_suffix(".rltracer.sqlite"))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(
            "CREATE TABLE IF NOT EXISTS indexed_steps(step INTEGER PRIMARY KEY, signature TEXT);"
            "CREATE TABLE IF NOT EXISTS trajectories(id INTEGER PRIMARY KEY, step INTEGER, prompt_key TEXT, preview TEXT, shard TEXT, row_i INTEGER);"
            "CREATE INDEX IF NOT EXISTS by_step_prompt ON trajectories(step, prompt_key, id);"
        )

    def list_steps(self, split: str | None = None) -> list[int]:
        return [0] if split in {None, "train", "eval"} else []

    def _iter_records(self) -> Iterator[tuple[int, dict[str, Any], list[dict[str, object]]]]:
        with self.jsonl_path.open(encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    messages = _messages(record)
                    if messages:
                        yield line_index, record, messages

    def ensure_step_indexed(self, step: int) -> None:
        if step != 0:
            raise KeyError(step)
        stat = self.jsonl_path.stat()
        signature = "workflow-jsonl-v1:" + hashlib.sha256(f"{self.jsonl_path}:{stat.st_mtime_ns}:{stat.st_size}".encode()).hexdigest()
        existing = self.db.execute("SELECT signature FROM indexed_steps WHERE step=?", (step,)).fetchone()
        if existing and existing["signature"] == signature:
            return
        self.db.execute("DELETE FROM trajectories WHERE step=?", (step,))
        for line_index, _record, messages in self._iter_records():
            key, preview = fingerprint(messages, self.prompt_parser)
            self.db.execute(
                "INSERT INTO trajectories(step,prompt_key,preview,shard,row_i) VALUES(?,?,?,?,?)",
                (step, key, preview, str(self.jsonl_path), line_index),
            )
        self.db.execute("INSERT OR REPLACE INTO indexed_steps VALUES(?,?)", (step, signature))
        self.db.commit()

    def list_prompts(self, step: int, split: str | None = None) -> list[tuple[str, int, str]]:
        self.ensure_step_indexed(step)
        rows = self.db.execute(
            "SELECT prompt_key,COUNT(*) n,MIN(preview) preview FROM trajectories WHERE step=? GROUP BY prompt_key ORDER BY prompt_key", (step,)
        )
        return [(row["prompt_key"], row["n"], row["preview"]) for row in rows]

    def list_trajectory_ids(self, step: int, prompt_key: str, split: str | None = None) -> list[int]:
        self.ensure_step_indexed(step)
        return [row["id"] for row in self.db.execute(
            "SELECT id FROM trajectories WHERE step=? AND prompt_key=? ORDER BY id", (step, prompt_key)
        )]

    def trajectory_split(self, trajectory_id: int) -> str | None:
        return None

    def load_trajectory(self, trajectory_id: int) -> Trajectory:
        row = self.db.execute("SELECT * FROM trajectories WHERE id=?", (trajectory_id,)).fetchone()
        if row is None:
            raise KeyError(trajectory_id)
        for line_index, record, messages in self._iter_records():
            if line_index == row["row_i"]:
                metadata = dict(row)
                metadata["trace"] = record
                metadata["input_format"] = "workflow-jsonl"
                return Trajectory(trajectory_id, 0, row["prompt_key"], messages, metadata)
        raise KeyError(f"Workflow JSONL line {row['row_i']} no longer exists in {self.jsonl_path}")

    def close(self) -> None:
        self.db.close()
