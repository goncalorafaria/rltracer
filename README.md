# RLTracer

`rltracer` is a lazy inspection layer for trajectories produced by different
reinforcement-learning training systems. It keeps library-specific file formats
inside adapters while exposing one common navigation model.

It is a standalone package: it does not import JTC, Datadev, JTCEval, or
PrimeBeaker. Those projects can consume its public adapters and exporters.
Generated traces, indexes, datasets, and plots are deliberately excluded from
the distribution and Git repository.

## Install

```bash
pip install rltracer
```

PrimeRL token exports require the adapter extra:

```bash
pip install 'rltracer[primerl]'
```

Use `rltracer[retemplate]` for Parquet-backed SFT retemplating,
`rltracer[plot]` for pass-rate plots, or `rltracer[all]` for every feature.
All command-line interfaces use Fire:

```bash
rltracer inspect --run=/path/to/run --tokenizer=/path/to/tokenizer
rltracer-viz --browse-root=/path/to/runs --tokenizer=/path/to/tokenizer
```

## Development and release

```bash
uv lock
uv run --extra test pytest
uv build
uv publish
```

The build produces a source distribution and universal wheel under `dist/`.
Configure a PyPI trusted publisher for the
[`goncalorafaria/rltracer`](https://github.com/goncalorafaria/rltracer)
repository before the first release; neither credentials nor generated
distributions are committed.

## Object hierarchy

```text
RLTracer
└── RolloutStep
    └── PromptGroup
        └── TrajectoryRef
            └── Trajectory
                └── messages
```

- **RLTracer** is the entry point for one training run. It receives a configured
  adapter in its constructor.
- **RolloutStep** represents one RL training step, such as `step_21`.
- **PromptGroup** represents all sampled trajectories sharing the same initial
  prompt within a rollout step.
- **TrajectoryRef** is a lightweight SQLite-backed reference. Calling
  `load()` decodes its source trajectory.
- **Trajectory** is the decoded multi-turn record. Its `messages` field is a
  normalized list of `{"role": ..., "content": ...}` dictionaries.

The canonical internal word is **trajectory**. A rollout is the sampled
trajectory produced by the RL system.

## Prime-RL adapter

`PrimeRLAdapter` currently supports Prime-RL rollout artifacts:

```text
<run>/rollouts/
  step_1/
    rank_0.bin
    rank_1.bin
  step_2/
    rank_0.bin
    ...
```

Each `rank_*.bin` file is a MessagePack payload. It contains groups of packed
Qwen token sequences. One packed sequence can contain multiple independent
conversations, so the adapter decodes it with the configured tokenizer and
splits it at each new Qwen system turn.

## Lazy behavior

Opening a tracer and listing its steps is cheap:

```python
adapter = PrimeRLAdapter(
    rollout_dir="/path/to/run/rollouts",
    tokenizer_name="/path/to/tokenizer",
    index_path="/tmp/prime-rl.sqlite",  # optional
)
tracer = RLTracer(adapter)

tracer.steps()  # discovers step directories only
```

No shard is read and no tokenizer is loaded until a prompt query is made:

```python
step = tracer.step(21)
prompt_groups = step.prompts()
```

The first `prompts()` call for a step:

1. Reads only that step's `rank_*.bin` files.
2. Decodes packed sequences with the adapter tokenizer.
3. Splits them into conversations.
4. Computes a stable hash of each conversation's pre-assistant prompt context.
5. Stores only lookup metadata in SQLite.

The SQLite index stores shard location, packed group/row location, conversation
position, prompt key, preview, and token count. It does not copy full token
sequences or decoded conversation content.

Loading a selected trajectory re-reads only the shard named in its SQLite row,
decodes the selected packed sequence, and returns the requested conversation:

```python
group = step.prompts()[0]
trajectory = group.trajectory(0).load()

for message in trajectory.messages:
    print(message["role"], message["content"])
```

The index is invalidated and rebuilt for a step if its shard paths, sizes, or
modification times change.

## Adapter boundary

New RL systems should implement the methods consumed by `RLTracer`:

```python
class Adapter:
    rollout_dir: Path

    def list_steps(self) -> list[int]: ...
    def ensure_step_indexed(self, step: int) -> None: ...
    def list_prompts(self, step: int) -> list[tuple[str, int, str]]: ...
    def list_trajectory_ids(self, step: int, prompt_key: str) -> list[int]: ...
    def load_trajectory(self, trajectory_id: int) -> Trajectory: ...
```

The adapter owns all source-specific choices: artifact format, tokenizer,
decoding, prompt grouping, optional rewards/logprobs, and index location.
`RLTracer` stays provider-neutral.

## Training-data export

TraceDataExporter builds JSONL directly from one or more structured PrimeRL
trace runs. A run can be an experiment directory, its run_default directory,
or its rollouts directory.

SFT mode selects complete trajectories:

```python
from rltracer import (
    Acceptance,
    DifficultyFilter,
    ExportPreferences,
    FieldFilter,
    TraceDataExporter,
    TraceRunSpec,
)

exporter = TraceDataExporter(
    runs=[
        TraceRunSpec("/path/to/first/run", step_end=800),
        TraceRunSpec("/path/to/second/run", step_end=400),
    ],
    preferences=ExportPreferences(
        mode="sft",
        acceptance=Acceptance.CORRECT,
        # Empirical prompt pass rate of 0.25-0.75.
        difficulty=DifficultyFilter(
            min_pass_rate=0.25,
            max_pass_rate=0.75,
        ),
        field_filters=(
            FieldFilter("metrics.num_turns", maximum=8),
            FieldFilter("rewards.reward.score", minimum=0),
        ),
        window_size=100,
        per_prompt=2,
        target_count=16_000,
        seed=17,
    ),
)
result = exporter.write_jsonl("correct_sft_16k.jsonl")
```

RL mode writes datadev RL rows whose prompt is the context preceding the first
assistant turn:

```python
rl_exporter = TraceDataExporter(
    [
        TraceRunSpec(
            "/path/to/run",
            step_end=800,
            rl_data_path="/path/to/datadev_rl_train.jsonl",
        )
    ],
    ExportPreferences(
        mode="rl",
        acceptance="any",
        difficulty=DifficultyFilter(min_difficulty=0.25),
        # Use one pass-rate estimate over the complete step range.
        window_size=None,
        unique_prompts=True,
        target_count=5_000,
    ),
)
rl_exporter.write_jsonl("rl_prompts_5k.jsonl")
```

Difficulty is identical to prompt_pass_rate. Pass rate is calculated from all scored
rollouts for a prompt in each configured step window. RL mode emits at most one
copy of a prompt per run/window; SFT mode honors per_prompt. Global targets are
deterministic and fair: every eligible prompt group receives its first row
before any group receives a second.

Set `unique_prompts=True` to keep at most one row for a prompt across all runs
and step windows. Deduplication is deterministic under `seed` and happens
before applying `target_count`.

Set `tool_call_sampling_power` to softly reduce the prominence of traces with
few command/tool calls. Selection weight is
`tool_call_count ** tool_call_sampling_power`, applied both within prompt
groups and when filling the global target. `0` preserves uniform selection,
`0.5` is a gentle square-root bias, `1` is proportional to command count, and
`2` is a strong quadratic bias. Selection remains deterministic under `seed`,
and export summaries include available and selected tool-call histograms.

By default, output rows match `datadev data` exactly. SFT rows contain
`messages` and optional JSON-string `tools`. RL rows are losslessly joined
through `task.data.idx` to `rl_data_path` and contain `prompt`, `answer`,
`output`, `feedback`, `record_id`, `rubric_index`, `source`, and list-valued
`tools`. Set `include_provenance=True` only when an additional
`trace_provenance` field is acceptable.

SFT tool calls are canonicalized to OpenAI's nested representation:
`{"id": ..., "type": "function", "function": {"name": ..., "arguments":
...}}`. Legacy flattened and singular calls are converted during export, and
calls without a function name are rejected rather than emitted as corrupted
training targets.

## Next extensions

- Add reward, advantage, log-probability, tool-call, and completion metadata to
  `Trajectory.metadata`.
