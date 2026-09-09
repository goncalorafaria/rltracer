"""Run the local RLTracer web explorer."""

from __future__ import annotations

import json
import os
import tempfile
from http import HTTPStatus
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import fire

from rltracer import PrimeRLAdapter, RLTracer, WorkflowJSONLAdapter
from rltracer.primerl import fingerprint

WEB_ROOT = Path(__file__).parent


def rollout_dir_for(root: Path) -> Path:
    """Return the PrimeRL export directory for a run or export directory."""
    if root.name in {"rollouts", "token_exports"}:
        return root
    rollouts = root / "rollouts"
    if rollouts.is_dir():
        return rollouts
    token_exports = root / "token_exports"
    return token_exports if token_exports.is_dir() else rollouts


def saved_steps(directory: Path) -> list[int]:
    """List saved rollout checkpoints without descending into trace files."""
    exports = rollout_dir_for(directory)
    return sorted(
        int(child.name[5:])
        for child in exports.glob("step_*")
        if child.is_dir() and child.name[5:].isdigit()
    )


def is_trace_run(directory: Path, input_format: str) -> bool:
    """Whether a path can be opened by the selected trace reader."""
    return directory.is_file() if input_format == "workflow-jsonl" else bool(saved_steps(directory))


class TraceRegistry:
    """Resolve browser-selected runs and keep their lazy indexes alive."""

    def __init__(self, tokenizer: str | None, default_run: Path | None, browse_root: Path | None, index_path: Path | None, input_format: str):
        self.tokenizer = tokenizer
        self.default_run = default_run.resolve() if default_run else None
        self.browse_root = browse_root.resolve() if browse_root else None
        self.index_path = index_path
        self.input_format = input_format
        self.tracers: dict[Path, RLTracer] = {}

    def _path_for(self, selected: str | None) -> Path:
        if selected:
            if self.browse_root is None:
                raise ValueError("This server was started without --browse-root")
            candidate = (self.browse_root / selected).resolve()
            if candidate != self.browse_root and self.browse_root not in candidate.parents:
                raise ValueError("Selected path is outside the browse root")
            return candidate
        if self.default_run is not None:
            return self.default_run
        raise ValueError("Choose a run from the browser")

    def tracer(self, selected: str | None) -> RLTracer:
        root = self._path_for(selected)
        if not is_trace_run(root, self.input_format):
            expected = "workflow JSONL" if self.input_format == "workflow-jsonl" else "PrimeRL trace output"
            raise ValueError(f"Not a {expected}: {root}")
        if root not in self.tracers:
            index_path = self.index_path if self.browse_root is None else None
            if index_path is None:
                descriptor, temporary_index = tempfile.mkstemp(prefix="rltracer-", suffix=".sqlite", dir="/tmp")
                os.close(descriptor)
                index_path = Path(temporary_index)
            if self.input_format == "workflow-jsonl":
                self.tracers[root] = RLTracer(WorkflowJSONLAdapter(root, index_path))
            else:
                if self.tokenizer is None:
                    raise ValueError("--tokenizer is required with --input-format primerl")
                self.tracers[root] = RLTracer(PrimeRLAdapter(rollout_dir_for(root), self.tokenizer, index_path))
        return self.tracers[root]

    def browse(self, selected: str = "") -> dict[str, object]:
        if self.browse_root is None:
            raise ValueError("This server was started without --browse-root")
        directory = (self.browse_root / selected).resolve()
        if directory != self.browse_root and self.browse_root not in directory.parents:
            raise ValueError("Selected path is outside the browse root")
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {selected}")
        children = []
        for child in sorted(directory.iterdir(), key=lambda item: item.name.casefold()):
            run_default = child / "run_default"
            if not child.is_dir() or child.name.startswith(".") or not run_default.is_dir():
                continue
            steps = saved_steps(run_default)
            if not steps:
                continue
            children.append({
                "name": child.name,
                "path": str(run_default.relative_to(self.browse_root)),
                "is_trace_run": True,
                "step_count": len(steps),
                "latest_step": steps[-1],
            })
        relative = directory.relative_to(self.browse_root)
        selected_path = "" if relative == Path(".") else str(relative)
        parent = None if directory == self.browse_root else str(directory.parent.relative_to(self.browse_root))
        return {"path": selected_path, "parent": parent, "is_trace_run": is_trace_run(directory, self.input_format), "children": children}

    def close(self) -> None:
        for tracer in self.tracers.values():
            tracer.close()


def send(handler: SimpleHTTPRequestHandler, data: object, status: int = 200) -> None:
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def step_prompt_pass_rates(adapter: PrimeRLAdapter | WorkflowJSONLAdapter, step: int, split: str | None) -> dict[str, dict[str, object]]:
    """Calculate saved final-label pass@1 once for every PrimeRL prompt."""
    if not isinstance(adapter, PrimeRLAdapter):
        return {}
    rates: dict[str, dict[str, object]] = {}
    for path in adapter._trace_files(step):
        if split and split not in path.parts:
            continue
        for _line_index, messages, record in adapter._iter_trace_file(path):
            metric = record.get("metrics", {}).get("correct_final_label")
            if not isinstance(metric, (int, float)):
                continue
            key, _preview = fingerprint(messages, adapter.prompt_parser)
            row = rates.setdefault(key, {"success_count": 0, "scored_rollout_count": 0})
            row["success_count"] = int(row["success_count"]) + int(metric >= 1.0)
            row["scored_rollout_count"] = int(row["scored_rollout_count"]) + 1
    for row in rates.values():
        row["pass_at_1"] = int(row["success_count"]) / int(row["scored_rollout_count"])
    return rates


def trace_outcome(trace: object) -> dict[str, object]:
    """Expose concise verifier outcomes without returning the full raw trace."""
    metadata = getattr(trace, "metadata", {})
    raw_trace = metadata.get("trace") if isinstance(metadata, dict) else None
    if not isinstance(raw_trace, dict):
        return {}
    return {
        key: raw_trace.get(key)
        for key in ("metrics", "rewards", "ok", "is_completed", "errors")
        if key in raw_trace
    }


def make_handler(registry: TraceRegistry):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            query = parse_qs(parsed.query)
            split = query.get("split", [None])[0]
            selected_run = query.get("run", [None])[0]
            if split not in {None, "train", "eval"}:
                return send(self, {"error": f"Unknown trace split {split!r}"}, HTTPStatus.BAD_REQUEST)
            try:
                if path in {"/", "/browse"} and registry.browse_root is not None:
                    self.path = "/browse.html"
                    return super().do_GET()
                if path == "/explorer":
                    self.path = "/index.html"
                    return super().do_GET()
                if not path.startswith("/api/"):
                    return super().do_GET()
                if path == "/api/runs":
                    return send(self, registry.browse(query.get("path", [""])[0]))
                tracer = registry.tracer(selected_run)
                if path == "/api/steps":
                    return send(self, [{"number": step} for step in tracer.adapter.list_steps(split)])
                if path.startswith("/api/steps/") and path.endswith("/prompts"):
                    step = tracer.step(int(path.split("/")[3]))
                    pass_rates = step_prompt_pass_rates(tracer.adapter, step.number, split)
                    return send(self, [
                        {
                            "key": group.key,
                            "rollout_count": group.rollout_count,
                            "preview": group.preview,
                            **pass_rates.get(group.key, {}),
                        }
                        for group in step.prompts(split)
                    ])
                if "/prompts/" in path and path.endswith("/trajectories"):
                    parts = path.split("/")
                    group = tracer.step(int(parts[3])).prompt(parts[5], split)
                    return send(
                        self,
                        [
                            {
                                "id": ref.id,
                                "split": tracer.adapter.trajectory_split(ref.id),
                                "correct_final_label": trace_outcome(ref.load()).get("metrics", {}).get("correct_final_label"),
                            }
                            for ref in group.trajectories(split)
                        ],
                    )
                if path.startswith("/api/trajectories/"):
                    trace = tracer.adapter.load_trajectory(int(path.rsplit("/", 1)[1]))
                    outcome = trace_outcome(trace)
                    metadata = {key: value for key, value in trace.metadata.items() if key != "trace"}
                    return send(self, {"id": trace.id, "step": trace.step, "messages": trace.messages, "metadata": metadata, "outcome": outcome})
                return super().do_GET()
            except (KeyError, ValueError, IndexError) as error:
                return send(self, {"error": str(error)}, HTTPStatus.NOT_FOUND)
            except Exception as error:
                return send(self, {"error": f"{type(error).__name__}: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    return Handler


def serve(
    run: str | None = None,
    browse_root: str | None = None,
    tokenizer: str | None = None,
    input_format: str = "primerl",
    index_path: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8787,
) -> None:
    """Serve one trace run or a browsable directory of PrimeRL runs."""
    if run is None and browse_root is None:
        raise ValueError("provide --run or --browse-root")
    if input_format not in {"primerl", "workflow-jsonl"}:
        raise ValueError("input_format must be primerl or workflow-jsonl")
    if input_format == "workflow-jsonl" and browse_root:
        raise ValueError(
            "--browse-root is not supported with workflow-jsonl; "
            "pass one JSONL file with --run"
        )
    if input_format == "primerl" and not tokenizer:
        raise ValueError("--tokenizer is required with --input-format primerl")
    run_path = Path(run) if run else None
    browse_path = Path(browse_root) if browse_root else None
    index = Path(index_path) if index_path else None
    registry = TraceRegistry(tokenizer, run_path, browse_path, index, input_format)
    landing = "/browse" if browse_path else "/explorer"
    print(f"RLTracer Explorer: http://{host}:{port}{landing}")
    if browse_path:
        print(f"Browse root: {browse_path.resolve()}")
    try:
        HTTPServer((host, port), make_handler(registry)).serve_forever()
    finally:
        registry.close()


def main() -> None:
    fire.Fire(serve)


if __name__ == "__main__":
    main()
