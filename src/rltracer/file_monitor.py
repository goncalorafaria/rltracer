"""Reader for PrimeRL file-monitor JSONL trace streams."""
from __future__ import annotations

import json
import mmap
import re
import shutil
import sqlite3
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .model import Trajectory
from .primerl import default_prompt_parser, fingerprint

_METRIC_STEP = re.compile(rb'"step"\s*:\s*(\d+)')


class FileMonitorAdapter:
    """Index and inspect JSONL (optionally zstd-compressed) monitor streams."""

    def __init__(self, run_dir: str | Path, index_path: str | Path | None = None):
        root = Path(run_dir).resolve()
        stream = root if root.name == "stream" else root / "monitors/file/traces/stream"
        if not stream.is_dir():
            matches = sorted(root.glob("*/monitors/file/traces/stream"))
            if len(matches) == 1:
                stream = matches[0]
        if not stream.is_dir():
            raise ValueError(f"No file-monitor trace stream under {root}")
        self.rollout_dir = stream
        self.prompt_parser = default_prompt_parser
        self._metric_steps_signature: tuple[int, int, int] | None = None
        self._metric_steps: set[int] = set()
        self._first_step_cache: dict[tuple[str, int, int, int], tuple[int | None, str | None]] = {}
        self.db = sqlite3.connect(index_path or stream / ".rltracer.sqlite", check_same_thread=False)
        self._metric_offset = 0
        self._files_cache = []
        self._files_checked = 0.0
        self.db.row_factory = sqlite3.Row
        self.db.executescript(
            "CREATE TABLE IF NOT EXISTS indexed_steps(step INTEGER PRIMARY KEY, signature TEXT);"
            "CREATE TABLE IF NOT EXISTS indexed_step_splits(step INTEGER, split TEXT, signature TEXT, PRIMARY KEY(step, split));"
            "CREATE TABLE IF NOT EXISTS trajectories(id INTEGER PRIMARY KEY, step INTEGER, prompt_key TEXT, preview TEXT, shard TEXT, row_i INTEGER, trace_id TEXT, byte_offset INTEGER, correct_final_label REAL, split TEXT);"
        )
        self.db.execute("CREATE TABLE IF NOT EXISTS chunk_heads(path TEXT PRIMARY KEY, inode INTEGER, first_step INTEGER, split TEXT)")
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(trajectories)")}
        if "byte_offset" not in columns:
            self.db.execute("ALTER TABLE trajectories ADD COLUMN byte_offset INTEGER")
        if "correct_final_label" not in columns:
            self.db.execute("ALTER TABLE trajectories ADD COLUMN correct_final_label REAL")
        if "split" not in columns:
            self.db.execute("ALTER TABLE trajectories ADD COLUMN split TEXT")
        self.db.execute("CREATE INDEX IF NOT EXISTS by_step_split_prompt ON trajectories(step, split, prompt_key, id)")
        # A per-file cursor replaces the old per-step, ordered-stream cache.
        self.db.execute("CREATE TABLE IF NOT EXISTS stream_cursors(path TEXT PRIMARY KEY, inode TEXT, size INTEGER, mtime_ns INTEGER, offset INTEGER)")
        self.db.execute("CREATE TABLE IF NOT EXISTS monitor_index_version(version INTEGER)")
        if self.db.execute("SELECT version FROM monitor_index_version WHERE version=5").fetchone() is None:
            self.db.execute("DELETE FROM trajectories")
            self.db.execute("DELETE FROM indexed_steps")
            self.db.execute("DELETE FROM indexed_step_splits")
            self.db.execute("DELETE FROM stream_cursors")
            self.db.execute("DELETE FROM monitor_index_version")
            self.db.execute("INSERT INTO monitor_index_version VALUES(5)")
        self.db.commit()


    def _files(self) -> list[Path]:
        if time.monotonic() - self._files_checked > 5:
            self._files_cache = sorted([*self.rollout_dir.glob("*.jsonl"), *self.rollout_dir.glob("*.jsonl.zst")])
            self._files_checked = time.monotonic()
        return list(self._files_cache)

    @staticmethod
    @contextmanager
    def _open(path: Path):
        if path.name.endswith(".zst"):
            try:
                import zstandard
            except ImportError:
                zstdcat = shutil.which("zstdcat")
                if zstdcat is None:
                    raise RuntimeError("Reading .zst monitor traces requires rltracer[file-monitor] or zstdcat")
                process = subprocess.Popen(
                    [zstdcat, str(path)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8"
                )
                try:
                    assert process.stdout is not None
                    yield process.stdout
                finally:
                    if process.stdout is not None:
                        process.stdout.close()
                    if process.poll() is None:
                        process.terminate()
                    process.wait()
                return
            with zstandard.open(path, "rt", encoding="utf-8") as handle:
                yield handle
            return
        with path.open(encoding="utf-8") as handle:
            yield handle

    def _records(self, path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
        with self._open(path) as handle:
            for row_i, line in enumerate(handle):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    yield row_i, row

    @staticmethod
    def _jsonl_records_with_offsets(path: Path) -> Iterator[tuple[int, int, dict[str, Any]]]:
        """Yield plain JSONL records with byte offsets for direct lazy reads."""
        with path.open("rb") as handle:
            row_i = 0
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    return
                try:
                    row = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    row_i += 1
                    continue
                if isinstance(row, dict):
                    yield row_i, offset, row
                row_i += 1

    @staticmethod
    def _reverse_jsonl_records(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
        """Read JSONL records newest-first, retaining direct byte offsets."""
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            if not size:
                return
            with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
                end = size
                while end > 0:
                    if data[end - 1:end] == b"\n":
                        end -= 1
                        if end == 0:
                            return
                    start = data.rfind(b"\n", 0, end) + 1
                    try:
                        record = json.loads(data[start:end])
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        end = start
                        continue
                    if isinstance(record, dict):
                        yield start, record
                    end = start

    @staticmethod
    def _trace_messages(trace: dict[str, Any]) -> list[dict[str, object]]:
        result = []
        for node in trace.get("nodes", []):
            message = node.get("message") if isinstance(node, dict) else None
            if not isinstance(message, dict) or not isinstance(message.get("role"), str):
                continue
            item = {key: value for key, value in message.items() if key != "role"}
            item["role"] = message["role"]
            item["content"] = str(item.get("content") or "")
            result.append(item)
        return result

    @staticmethod
    def _step(record: dict[str, Any], trace: dict[str, Any]) -> int | None:
        info = trace.get("info") or {}
        dispatch = info.get("dispatch") or {}
        work = (record.get("run") or {}).get("work") or {}
        value = dispatch.get("step", work.get("step"))
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    @staticmethod
    def _split(record: dict[str, Any]) -> str | None:
        """Return the rollout split from PrimeRL's work metadata."""
        work = (record.get("run") or {}).get("work") or {}
        value = work.get("type")
        if value in {"train", "eval"}:
            return value
        environment = (record.get("env") or {}).get("name")
        if isinstance(environment, str):
            if environment.endswith("-train"):
                return "train"
            if environment.endswith("-eval"):
                return "eval"
        return None

    def _first_step(self, path: Path) -> tuple[int | None, str | None]:
        stat = path.stat()
        # The first complete record cannot change in append-only monitor files.
        signature = (str(path), hex(stat.st_ino))
        if signature not in self._first_step_cache:
            saved = self.db.execute("SELECT first_step,split FROM chunk_heads WHERE path=? AND inode=?", signature).fetchone()
            if saved is not None and stat.st_size:
                self._first_step_cache[signature] = (saved[0], saved[1])
            else:
                first = next(self._records(path), None)
                steps = [self._step(first[1], trace) for trace in first[1].get("traces", []) if isinstance(trace, dict)] if first else []
                steps = [step for step in steps if step is not None]
                value = (min(steps) if steps else None, self._split(first[1]) if first else None)
                if first:
                    self._first_step_cache[signature] = value
                    self.db.execute("INSERT OR REPLACE INTO chunk_heads VALUES(?,?,?,?)", (*signature, *value))
                    self.db.commit()
                return value
        return self._first_step_cache[signature]

    def _steps_from_metrics(self) -> set[int]:
        path = self.rollout_dir.parent.parent / "metrics.jsonl"
        if not path.is_file():
            return set()
        stat = path.stat()
        signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if signature != self._metric_steps_signature:
            if self._metric_steps_signature is None or stat.st_ino != self._metric_steps_signature[0] or stat.st_size < self._metric_offset:
                self._metric_steps.clear()
                self._metric_offset = 0
            if stat.st_size:
                with path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
                    # Keep an incomplete final line for the next refresh.
                    end = data.rfind(b"\n", self._metric_offset) + 1
                    if end > self._metric_offset:
                        self._metric_steps.update(int(m.group(1)) for m in _METRIC_STEP.finditer(data, self._metric_offset, end))
                        self._metric_offset = end
            self._metric_steps_signature = signature
        return set(self._metric_steps)

    def _latest_plain_trace_step(self) -> int | None:
        """Read the final JSONL record so the picker defaults to the live step."""
        paths = [path for path in self._files() if path.name.endswith(".jsonl")]
        if not paths:
            return None
        path = paths[-1]
        with path.open("rb") as handle:
            handle.seek(0, 2)
            end = handle.tell()
            if end == 0:
                return None
            handle.seek(end - 1)
            if handle.read(1) == b"\n":
                end -= 1
            line_start = 0
            cursor = end
            while cursor > 0:
                start = max(0, cursor - 64 * 1024)
                handle.seek(start)
                block = handle.read(cursor - start)
                separator = block.rfind(b"\n")
                if separator >= 0:
                    line_start = start + separator + 1
                    break
                cursor = start
            handle.seek(line_start)
            try:
                record = json.loads(handle.readline())
            except (json.JSONDecodeError, UnicodeDecodeError):
                return None
        steps = [
            step
            for trace in record.get("traces", [])
            if isinstance(trace, dict)
            if (step := self._step(record, trace)) is not None
        ]
        return max(steps) if steps else None

    def list_steps(self, split: str | None = None) -> list[int]:
        self._sync_streams()
        where = " WHERE split=?" if split is not None else ""
        params = (split,) if split is not None else ()
        return [row[0] for row in self.db.execute(
            "SELECT DISTINCT step FROM trajectories" + where + " ORDER BY step", params
        )]

    def _index_record(self, path: Path, row_i: int, record: dict[str, Any], byte_offset: int | None) -> None:
        record_split = self._split(record)
        for trace in record.get("traces", []):
            if not isinstance(trace, dict):
                continue
            step = self._step(record, trace)
            if step is None:
                continue
            messages = self._trace_messages(trace)
            if not messages:
                continue
            key, preview = fingerprint(messages, self.prompt_parser)
            correct = (trace.get("metrics") or {}).get("correct_final_label")
            if isinstance(correct, bool):
                correct = float(correct)
            elif not isinstance(correct, (int, float)):
                correct = None
            self.db.execute(
                "INSERT INTO trajectories(step,prompt_key,preview,shard,row_i,trace_id,byte_offset,correct_final_label,split) VALUES(?,?,?,?,?,?,?,?,?)",
                (step, key, preview, str(path), row_i, str(trace.get("id", "")), byte_offset, correct, record_split),
            )

    def _sync_streams(self) -> None:
        # Rollouts finish asynchronously: any chunk can contain any step/split.
        # Index all completed records once, then consume only appended bytes.
        deadline = time.monotonic() + 2.0
        files = self._files()
        paths = {str(path) for path in files}
        with self.db:
            for row in self.db.execute("SELECT path FROM stream_cursors").fetchall():
                if row[0] not in paths:
                    self.db.execute("DELETE FROM trajectories WHERE shard=?", (row[0],))
                    self.db.execute("DELETE FROM stream_cursors WHERE path=?", (row[0],))
            for path in files:
                stat = path.stat()
                inode = str(stat.st_ino)
                saved = self.db.execute("SELECT * FROM stream_cursors WHERE path=?", (str(path),)).fetchone()
                if saved and saved["offset"] >= stat.st_size and (saved["inode"], saved["size"], saved["mtime_ns"]) == (inode, stat.st_size, stat.st_mtime_ns):
                    continue
                plain = path.name.endswith(".jsonl")
                append = saved and plain and saved["inode"] == inode and (stat.st_size > saved["size"] or (stat.st_size == saved["size"] and stat.st_mtime_ns == saved["mtime_ns"]))
                offset = saved["offset"] if append else 0
                if not append:
                    self.db.execute("DELETE FROM trajectories WHERE shard=?", (str(path),))
                if plain:
                    with path.open("rb") as handle:
                        handle.seek(offset)
                        # Snapshot size avoids chasing an actively growing file.
                        while handle.tell() < stat.st_size:
                            start = handle.tell()
                            line = handle.readline(stat.st_size - start)
                            if not line.endswith(b"\n"):
                                break  # Retry this unfinished record after append.
                            offset = handle.tell()
                            try:
                                record = json.loads(line)
                            except (json.JSONDecodeError, UnicodeDecodeError):
                                continue
                            if isinstance(record, dict):
                                self._index_record(path, start, record, start)
                            if time.monotonic() >= deadline:
                                break
                else:
                    # Compressed chunks cannot be sought by uncompressed offset.
                    for row_i, record in self._records(path):
                        self._index_record(path, row_i, record, None)
                    offset = stat.st_size
                self.db.execute(
                    "INSERT OR REPLACE INTO stream_cursors VALUES(?,?,?,?,?)",
                    (str(path), inode, stat.st_size, stat.st_mtime_ns, offset),
                )
                if time.monotonic() >= deadline:
                    break

    def ensure_step_indexed(self, step: int, split: str | None = None) -> None:
        self._sync_streams()

    def list_prompts(self, step: int, split: str | None = None) -> list[tuple[str, int, str]]:
        self.ensure_step_indexed(step, split)
        where, params = "step=?", [step]
        if split:
            where += " AND split=?"
            params.append(split)
        return [(r["prompt_key"], r["n"], r["preview"]) for r in self.db.execute(f"SELECT prompt_key,COUNT(*) n,MIN(preview) preview FROM trajectories WHERE {where} GROUP BY prompt_key ORDER BY prompt_key", params)]

    def list_trajectory_ids(self, step: int, prompt_key: str, split: str | None = None) -> list[int]:
        self.ensure_step_indexed(step, split)
        where, params = "step=? AND prompt_key=?", [step, prompt_key]
        if split:
            where += " AND split=?"
            params.append(split)
        return [r[0] for r in self.db.execute(f"SELECT id FROM trajectories WHERE {where} ORDER BY id", params)]

    def list_trajectory_summaries(self, step: int, prompt_key: str, split: str | None = None) -> list[dict[str, Any]]:
        self.ensure_step_indexed(step, split)
        where, params = "step=? AND prompt_key=?", [step, prompt_key]
        if split:
            where += " AND split=?"
            params.append(split)
        return [
            {"id": row["id"], "split": row["split"], "correct_final_label": row["correct_final_label"]}
            for row in self.db.execute(
                f"SELECT id,correct_final_label,split FROM trajectories WHERE {where} ORDER BY id",
                params,
            )
        ]

    def trajectory_split(self, trajectory_id: int) -> str | None:
        row = self.db.execute("SELECT split FROM trajectories WHERE id=?", (trajectory_id,)).fetchone()
        return row["split"] if row else None

    def _backfill_byte_offsets(self, shard: Path, through_row: int) -> None:
        """Backfill offsets with a byte scan, without decoding earlier JSON records."""
        missing_rows = {
            row[0]
            for row in self.db.execute(
                "SELECT DISTINCT row_i FROM trajectories WHERE shard=? AND row_i<=? AND byte_offset IS NULL",
                (str(shard), through_row),
            )
        }
        if not missing_rows or not shard.name.endswith(".jsonl"):
            return
        last_needed = max(missing_rows)
        offsets: dict[int, int] = {}
        with shard.open("rb") as handle:
            if not handle.seek(0, 2):
                return
            size = handle.tell()
            if size == 0:
                return
            with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
                row_i = 0
                start = 0
                while row_i <= last_needed:
                    if row_i in missing_rows:
                        offsets[row_i] = start
                    newline = data.find(b"\n", start)
                    if newline < 0:
                        break
                    start = newline + 1
                    row_i += 1
        self.db.executemany(
            "UPDATE trajectories SET byte_offset=? WHERE shard=? AND row_i=? AND byte_offset IS NULL",
            [(offset, str(shard), row_i) for row_i, offset in offsets.items()],
        )
        self.db.commit()

    def load_trajectory(self, trajectory_id: int) -> Trajectory:
        row = self.db.execute("SELECT * FROM trajectories WHERE id=?", (trajectory_id,)).fetchone()
        if row is None:
            raise KeyError(trajectory_id)
        shard = Path(row["shard"])
        if row["byte_offset"] is None and shard.name.endswith(".jsonl"):
            self._backfill_byte_offsets(shard, row["row_i"])
            row = self.db.execute("SELECT * FROM trajectories WHERE id=?", (trajectory_id,)).fetchone()
        if row["byte_offset"] is not None and shard.name.endswith(".jsonl"):
            with shard.open("rb") as handle:
                handle.seek(row["byte_offset"])
                record = json.loads(handle.readline())
            records = [(row["row_i"], record)]
        else:
            records = self._records(shard)
        for row_i, record in records:
            if row_i != row["row_i"]:
                continue
            for trace in record.get("traces", []):
                if isinstance(trace, dict) and str(trace.get("id", "")) == row["trace_id"]:
                    task = trace.get("task") or {}
                    data = task.get("data") or {}
                    agent = trace.get("agent") or {}
                    config = agent.get("config") or {}
                    metadata = {
                        "trace": trace,
                        "run_record_id": record.get("id"),
                        "input_format": "file-monitor",
                        "environment": record.get("env", {}).get("name"),
                        "environment_id": record.get("env", {}).get("id"),
                        "task_type": task.get("type"),
                        "task_index": data.get("idx"),
                        "task_prompt": data.get("prompt"),
                        "judge_input": data.get("answer"),
                        "model": config.get("model"),
                        "available_tools": trace.get("tools", []),
                        "calls": trace.get("calls", []),
                        "num_input_tokens": trace.get("num_input_tokens"),
                        "num_output_tokens": trace.get("num_output_tokens"),
                        "judge_audit": (trace.get("info") or {}).get("judge"),
                    }
                    return Trajectory(trajectory_id, row["step"], row["prompt_key"], self._trace_messages(trace), metadata)
        raise KeyError(trajectory_id)

    def close(self) -> None:
        self.db.close()
