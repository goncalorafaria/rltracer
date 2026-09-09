from __future__ import annotations

import json
from pathlib import Path

import pytest

from rltracer.criteria_suffix import SearchCriteriaSuffixAugmenter


def _row(prompt: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "system"},
            {
                "role": "user",
                "content": (
                    f"<input>\n{prompt}\n</input>\n\n"
                    "<evaluation_criteria>\ncriterion\n"
                    "</evaluation_criteria>\n\n# SCORING RUBRIC\n\n"
                    "pass - yes\nfail - no"
                ),
            },
            {"role": "assistant", "content": answer},
        ],
        "tools": "[]",
    }


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_augments_only_search_rows_inside_evaluation_criteria(
    tmp_path: Path,
) -> None:
    search = _row("shared", "search answer")
    base = _row("base", "base answer")
    aligned = tmp_path / "aligned.jsonl"
    replay = tmp_path / "replay.jsonl"
    output = tmp_path / "output.jsonl"
    _write(aligned, [search, base])
    replay_rows = []
    for row, run in [(search, "search-quokka"), (base, "base-quokka")]:
        old = json.loads(json.dumps(row))
        old["messages"][1]["content"] = "old prompt"
        old["trace_provenance"] = {"run": run}
        replay_rows.append(old)
    _write(replay, replay_rows)
    augmenter = SearchCriteriaSuffixAugmenter(
        aligned,
        replay,
        ["check source one", "check source two"],
        run_name_substrings=["search-quokka"],
    )

    report = augmenter.augment_jsonl(aligned, output)
    rows = [json.loads(line) for line in output.read_text().splitlines()]

    assert report["criteria_augmented_rows"] == 1
    assert report["unchanged_rows"] == 1
    assert "criterion\n\ncheck source" in rows[0]["messages"][1]["content"]
    assert rows[1] == base
    assert rows[0]["messages"][2:] == search["messages"][2:]
    assert set(rows[0]) == {"messages", "tools"}


def test_rejects_misaligned_provenance_replay(tmp_path: Path) -> None:
    aligned = tmp_path / "aligned.jsonl"
    replay = tmp_path / "replay.jsonl"
    clean = _row("prompt", "answer")
    wrong = _row("prompt", "different answer")
    wrong["trace_provenance"] = {"run": "search-quokka"}
    _write(aligned, [clean])
    _write(replay, [wrong])
    augmenter = SearchCriteriaSuffixAugmenter(
        aligned,
        replay,
        ["check sources"],
        run_name_substrings=["search-quokka"],
    )

    with pytest.raises(ValueError, match="not aligned"):
        augmenter.augment_row(clean)


def test_does_not_duplicate_an_existing_source_check(tmp_path: Path) -> None:
    suffix = "check the underlying sources"
    clean = _row("prompt", "answer")
    clean["messages"][1]["content"] = clean["messages"][1][
        "content"
    ].replace("criterion", suffix)
    aligned = tmp_path / "aligned.jsonl"
    replay = tmp_path / "replay.jsonl"
    _write(aligned, [clean])
    old = json.loads(json.dumps(clean))
    old["messages"][1]["content"] = "old prompt"
    old["trace_provenance"] = {"run": "search-quokka"}
    _write(replay, [old])
    output = tmp_path / "output.jsonl"
    augmenter = SearchCriteriaSuffixAugmenter(
        aligned,
        replay,
        [suffix],
        run_name_substrings=["search-quokka"],
    )

    report = augmenter.augment_jsonl(aligned, output)

    assert report["criteria_augmented_rows"] == 0
    assert report["criteria_already_source_checking_rows"] == 1
    assert json.loads(output.read_text()) == clean
