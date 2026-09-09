import json
import shutil
from collections import Counter
from pathlib import Path

import pytest

from rltracer.export import (
    Acceptance,
    DifficultyFilter,
    ExportMode,
    ExportPreferences,
    FieldFilter,
    TraceDataExporter,
    TraceRunSpec,
)


OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "inspect",
            "parameters": {"type": "object"},
        },
    }
]


def trace(
    prompt: str,
    answer: str,
    *,
    correct: int,
    identifier: str,
    data_index: int,
    reward: float = 1.0,
    turns: int = 1,
) -> dict:
    sampled_nodes = []
    for turn_index in range(turns - 1):
        sampled_nodes.extend(
            [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": (
                                    f"call-{identifier}-{turn_index}"
                                ),
                                "name": "terminal",
                                "arguments": {"command": "inspect"},
                            }
                        ],
                    }
                },
                {
                    "message": {
                        "role": "tool",
                        "tool_call_id": (
                            f"call-{identifier}-{turn_index}"
                        ),
                        "content": "evidence",
                    }
                },
            ]
        )
    return {
        "id": identifier,
        "nodes": [
            {"message": {"role": "system", "content": "system"}},
            {"message": {"role": "user", "content": prompt}},
            *sampled_nodes,
            {
                "message": {
                    "role": "assistant",
                    "content": answer,
                    "tool_calls": [
                        {
                            "id": f"call-{identifier}",
                            "name": "terminal",
                            "arguments": {"command": "inspect"},
                        }
                    ],
                }
            },
        ],
        "tools": [
            {
                "name": "terminal",
                "description": "inspect",
                "parameters": {"type": "object"},
            }
        ],
        "task": {
            "data": {
                "idx": data_index,
                "answer": "pass" if correct else "fail",
            }
        },
        "metrics": {
            "correct_final_label": correct,
            "num_turns": turns,
        },
        "rewards": {"reward": {"score": reward}},
        "info": {
            "episode_id": f"episode-{identifier}",
            "policy_version": 1,
        },
    }


def write_run(tmp_path: Path) -> tuple[Path, Path]:
    experiment = tmp_path / "experiment"
    trace_path = (
        experiment
        / "run_default"
        / "rollouts"
        / "step_1"
        / "train"
        / "all"
        / "traces.jsonl"
    )
    trace_path.parent.mkdir(parents=True)
    specs = [
        ("medium", "m-correct-1", 1, "m1", 4, 3),
        ("medium", "m-correct-2", 1, "m2", 3, 2),
        ("medium", "m-wrong-1", 0, "m3", -3, 1),
        ("medium", "m-wrong-2", 0, "m4", -4, 1),
        ("easy", "e-correct-1", 1, "e1", 1, 1),
        ("easy", "e-correct-2", 1, "e2", 1, 1),
        ("hard", "h-wrong-1", 0, "h1", -1, 1),
        ("hard", "h-wrong-2", 0, "h2", -1, 1),
    ]
    traces = []
    data_rows = []
    for index, (
        prompt,
        assistant,
        correct,
        identifier,
        reward,
        turns,
    ) in enumerate(specs):
        traces.append(
            trace(
                prompt,
                assistant,
                correct=correct,
                identifier=identifier,
                data_index=index,
                reward=reward,
                turns=turns,
            )
        )
        data_rows.append(
            {
                "prompt": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": prompt},
                ],
                "answer": "pass" if correct else "fail",
                "output": f"hidden-output-{identifier}",
                "feedback": f"feedback-{identifier}",
                "record_id": f"record-{identifier}",
                "rubric_index": index,
                "source": "synthetic",
                "tools": OPENAI_TOOLS,
            }
        )
    trace_path.write_text(
        "".join(json.dumps(record) + "\n" for record in traces),
        encoding="utf-8",
    )
    data_path = tmp_path / "rldata_train.jsonl"
    data_path.write_text(
        "".join(json.dumps(row) + "\n" for row in data_rows),
        encoding="utf-8",
    )
    return experiment, data_path


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_sft_mode_matches_datadev_schema(
    tmp_path: Path,
) -> None:
    run, _data_path = write_run(tmp_path)
    output = tmp_path / "sft.jsonl"
    exporter = TraceDataExporter(
        [TraceRunSpec(run, step_end=1)],
        ExportPreferences(
            mode=ExportMode.SFT,
            acceptance=Acceptance.CORRECT,
            difficulty=DifficultyFilter(
                min_difficulty=0.25,
                max_difficulty=0.75,
            ),
            field_filters=(
                FieldFilter("rewards.reward.score", minimum=3),
            ),
            per_prompt=2,
        ),
    )

    result = exporter.write_jsonl(output)
    rows = read_jsonl(output)

    assert result.selected_count == 2
    assert result.available_count == 2
    assert len(rows) == 2
    assert all(set(row) == {"messages", "tools"} for row in rows)
    assert {
        row["messages"][-1]["content"]
        for row in rows
    } == {"m-correct-1", "m-correct-2"}
    assert all(isinstance(row["tools"], str) for row in rows)
    assert json.loads(rows[0]["tools"]) == OPENAI_TOOLS
    arguments = rows[0]["messages"][-1]["tool_calls"][0][
        "function"
    ]["arguments"]
    assert isinstance(arguments, str)
    assert json.loads(arguments) == {"command": "inspect"}

    summary = json.loads(result.summary_path.read_text())
    assert summary["preferences"]["mode"] == "sft"
    assert summary["eligible_prompt_groups"] == 1


def test_rl_mode_rejoins_exact_datadev_rows(
    tmp_path: Path,
) -> None:
    run, data_path = write_run(tmp_path)
    output = tmp_path / "rl.jsonl"
    exporter = TraceDataExporter(
        [TraceRunSpec(run, rl_data_path=data_path)],
        ExportPreferences(
            mode="rl",
            acceptance="any",
            per_prompt=99,
        ),
    )

    result = exporter.write_jsonl(output)
    rows = read_jsonl(output)

    expected_keys = {
        "prompt",
        "answer",
        "output",
        "feedback",
        "record_id",
        "rubric_index",
        "source",
        "tools",
    }
    assert result.selected_count == 3
    assert len(rows) == 3
    assert all(set(row) == expected_keys for row in rows)
    assert all(
        [message["role"] for message in row["prompt"]]
        == ["system", "user"]
        for row in rows
    )
    assert all(isinstance(row["tools"], list) for row in rows)
    assert all(row["tools"] == OPENAI_TOOLS for row in rows)
    assert all(row["output"].startswith("hidden-output-") for row in rows)
    assert all(row["record_id"].startswith("record-") for row in rows)


def test_sft_singular_tool_call_matches_datadev_normalization() -> None:
    messages = [
        {
            "role": "assistant",
            "tool_call": {
                "id": "legacy-call",
                "name": "terminal",
                "arguments": {
                    "command": ["head", "-n", 20],
                    "timeout": 10,
                },
            },
        }
    ]

    normalized = TraceDataExporter._sft_messages(messages)

    assert "tool_call" not in normalized[0]
    assert normalized[0]["tool_calls"] == [
        {
            "id": "legacy-call",
            "type": "function",
            "function": {
                "name": "terminal",
                "arguments": (
                    '{"command": ["head", "-n", 20], '
                    '"timeout": 10}'
                ),
            },
        }
    ]
    assert json.loads(
        normalized[0]["tool_calls"][0]["function"][
            "arguments"
        ]
    ) == {
        "command": ["head", "-n", 20],
        "timeout": 10,
    }
    assert messages[0]["tool_call"]["arguments"]["command"] == [
        "head",
        "-n",
        20,
    ]


def test_sft_tool_call_without_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="no function name"):
        TraceDataExporter._sft_messages(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "broken", "arguments": "{}"}
                    ],
                }
            ]
        )


def test_rl_difficulty_filter_uses_pass_rate(
    tmp_path: Path,
) -> None:
    run, data_path = write_run(tmp_path)
    output = tmp_path / "rl-medium.jsonl"
    exporter = TraceDataExporter(
        [TraceRunSpec(run, rl_data_path=data_path)],
        ExportPreferences(
            mode="rl",
            acceptance="any",
            difficulty=DifficultyFilter(
                min_difficulty=0.25,
                max_difficulty=0.75,
            ),
        ),
    )

    result = exporter.write_jsonl(output)
    rows = read_jsonl(output)

    assert result.selected_count == 1
    assert rows[0]["prompt"][-1]["content"] == "medium"


def test_unique_prompts_deduplicates_across_windows(
    tmp_path: Path,
) -> None:
    run, data_path = write_run(tmp_path)
    rollouts = run / "run_default" / "rollouts"
    shutil.copytree(rollouts / "step_1", rollouts / "step_2")
    output = tmp_path / "rl-unique.jsonl"
    exporter = TraceDataExporter(
        [TraceRunSpec(run, step_end=2, rl_data_path=data_path)],
        ExportPreferences(
            mode="rl",
            acceptance="any",
            window_size=1,
            unique_prompts=True,
        ),
    )

    result = exporter.write_jsonl(output)
    prompts = [
        json.dumps(row["prompt"], sort_keys=True)
        for row in read_jsonl(output)
    ]

    assert result.available_count == 6
    assert result.selected_count == 3
    assert len(prompts) == len(set(prompts)) == 3
    summary = json.loads(result.summary_path.read_text())
    assert summary["preferences"]["unique_prompts"] is True
    assert summary["eligible_unique_prompts"] == 3


def test_global_per_prompt_caps_across_runs_and_layers_fairly(
    tmp_path: Path,
) -> None:
    run_one, _ = write_run(tmp_path / "one")
    run_two, _ = write_run(tmp_path / "two")
    output = tmp_path / "globally-capped.jsonl"
    exporter = TraceDataExporter(
        [run_one, run_two],
        ExportPreferences(
            mode="sft",
            acceptance="any",
            per_prompt=2,
            global_per_prompt=2,
            target_count=4,
        ),
    )

    result = exporter.write_jsonl(output)
    prompt_counts = Counter(
        row["messages"][1]["content"]
        for row in read_jsonl(output)
    )

    assert result.available_count == 12
    assert result.selected_count == 4
    assert set(prompt_counts) == {"medium", "easy", "hard"}
    assert max(prompt_counts.values()) == 2
    summary = json.loads(result.summary_path.read_text())
    assert summary["preferences"]["global_per_prompt"] == 2


def test_inspect_applies_exact_fair_target(
    tmp_path: Path,
) -> None:
    run, _data_path = write_run(tmp_path)
    exporter = TraceDataExporter(
        [run],
        ExportPreferences(
            mode="sft",
            acceptance="any",
            per_prompt=2,
            target_count=4,
        ),
    )

    summary = exporter.inspect()

    assert summary["available_count"] == 6
    assert summary["selected_count"] == 4
    assert summary["eligible_prompt_groups"] == 3


def test_tool_call_sampling_power_favors_more_commands(
    tmp_path: Path,
) -> None:
    run, _data_path = write_run(tmp_path)
    trace_path = (
        run / "run_default" / "rollouts" / "step_1"
        / "train" / "all" / "traces.jsonl"
    )
    traces = read_jsonl(trace_path)
    target = next(row for row in traces if row["id"] == "e1")
    calls = target["nodes"][-1]["message"]["tool_calls"]
    for index in range(10):
        calls.append(
            {
                "id": f"call-e1-extra-{index}",
                "name": "terminal",
                "arguments": {"command": "inspect"},
            }
        )
    trace_path.write_text(
        "".join(json.dumps(row) + "\n" for row in traces),
        encoding="utf-8",
    )
    output = tmp_path / "longer.jsonl"
    exporter = TraceDataExporter(
        [run],
        ExportPreferences(
            mode="sft",
            acceptance="any",
            per_prompt=1,
            target_count=1,
            tool_call_sampling_power=100,
        ),
    )

    result = exporter.write_jsonl(output)
    row = read_jsonl(output)[0]

    assert result.selected_count == 1
    assert sum(
        message["role"] == "assistant"
        for message in row["messages"]
    ) == 1
    assert sum(
        len(message.get("tool_calls") or [])
        for message in row["messages"]
    ) == 11
    summary = json.loads(result.summary_path.read_text())
    assert summary["preferences"]["tool_call_sampling_power"] == 100
    assert summary["selected_tool_call_distribution"][
        "counts"
    ] == {"11": 1}


def test_target_rejects_shortfall(tmp_path: Path) -> None:
    run, data_path = write_run(tmp_path)
    exporter = TraceDataExporter(
        [TraceRunSpec(run, rl_data_path=data_path)],
        ExportPreferences(mode="rl", target_count=4),
    )

    with pytest.raises(ValueError, match="only 2 are available"):
        exporter.inspect()


def test_rl_write_requires_datadev_source(tmp_path: Path) -> None:
    run, _data_path = write_run(tmp_path)
    exporter = TraceDataExporter(
        [run],
        ExportPreferences(mode="rl", acceptance="any"),
    )

    with pytest.raises(
        ValueError,
        match="requires TraceRunSpec.rl_data_path",
    ):
        exporter.write_jsonl(tmp_path / "missing.jsonl")


def test_difficulty_is_the_pass_rate() -> None:
    assert DifficultyFilter(min_difficulty=0.75).accepts(0.8)
    assert not DifficultyFilter(min_difficulty=0.75).accepts(0.7)
    assert DifficultyFilter(max_difficulty=0.25).accepts(0.2)
    assert not DifficultyFilter(max_difficulty=0.25).accepts(0.3)


def test_invalid_preferences_are_rejected() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        DifficultyFilter(max_pass_rate=2)
    with pytest.raises(ValueError, match="positive"):
        ExportPreferences(per_prompt=0)
    with pytest.raises(ValueError, match="positive"):
        ExportPreferences(global_per_prompt=0)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        ExportPreferences(tool_call_sampling_power=-1)
    with pytest.raises(ValueError, match="must not exceed"):
        FieldFilter("metrics.num_turns", minimum=3, maximum=2)
