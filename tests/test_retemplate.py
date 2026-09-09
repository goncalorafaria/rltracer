from __future__ import annotations

import json
from pathlib import Path

import pytest

from rltracer.retemplate import MixTemplateSource, SFTTemplateRewriter


def _template(path: Path, *, tagged: bool) -> None:
    user = (
        "<input>\n{inputs}\n</input>\n\n"
        "<evaluation_criteria>\n{evaluation_criteria}\n</evaluation_criteria>\n\n"
        "# SCORING RUBRIC\n\n{rubric}"
        if tagged
        else "INPUT={inputs}\nCRITERION={evaluation_criteria}\nLABELS={rubric}"
    )
    path.write_text(
        json.dumps(
            {
                "input_variables": ["inputs", "evaluation_criteria", "rubric"],
                "tools": [
                    {
                        "name": "terminal",
                        "description": "browse only",
                        "parameters": {"type": "object"},
                    }
                ],
                "chat_template": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": user},
                ],
            }
        ),
        encoding="utf-8",
    )


def test_rewrites_prefix_and_tools_from_selected_jrow(tmp_path: Path) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    source_template = tmp_path / "source.json"
    target_template = tmp_path / "target.json"
    _template(source_template, tagged=False)
    _template(target_template, tagged=True)
    jrow = {
        "input": "original request",
        "output": "evaluated output",
        "rubric": {
            "criteria": "inspect the cited source",
            # Mirrors selected-jrows parquet's sort_keys=True serialization.
            "scores": {"fail": "unsupported", "pass": "supported"},
        },
    }
    mix_path = tmp_path / "selected.jrows.parquet"
    parquet.write_table(
        pyarrow.table({"jrow_json": [json.dumps(jrow)]}), mix_path
    )
    source_user = (
        "INPUT=original request\nCRITERION=inspect the cited source\n"
        "LABELS=pass - supported\nfail - unsupported"
    )
    row = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": source_user},
            {"role": "assistant", "content": "", "tool_calls": []},
            {"role": "tool", "content": "evidence"},
            {"role": "assistant", "content": '{"label":"pass"}'},
        ],
        "tools": "[]",
    }
    rewriter = SFTTemplateRewriter(
        [MixTemplateSource(mix_path, source_template)], target_template
    )

    rewritten = rewriter.rewrite_row(row)

    assert rewritten["messages"][:2] == [
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": (
                "<input>\noriginal request\n</input>\n\n"
                "<evaluation_criteria>\ninspect the cited source\n"
                "</evaluation_criteria>\n\n# SCORING RUBRIC\n\n"
                "pass - supported\nfail - unsupported"
            ),
        },
    ]
    assert rewritten["messages"][2:] == row["messages"][2:]
    tools = json.loads(rewritten["tools"])
    assert tools[0]["function"]["description"] == "browse only"


def test_unmatched_prefix_is_rejected(tmp_path: Path) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    source_template = tmp_path / "source.json"
    target_template = tmp_path / "target.json"
    _template(source_template, tagged=False)
    _template(target_template, tagged=True)
    mix_path = tmp_path / "selected.jrows.parquet"
    parquet.write_table(
        pyarrow.table(
            {
                "jrow_json": [
                    json.dumps(
                        {
                            "input": "known",
                            "rubric": {
                                "criteria": "criterion",
                                "scores": {"pass": "yes", "fail": "no"},
                            },
                        }
                    )
                ]
            }
        ),
        mix_path,
    )
    rewriter = SFTTemplateRewriter(
        [MixTemplateSource(mix_path, source_template)], target_template
    )

    with pytest.raises(ValueError, match="no selected-JRow match"):
        rewriter.rewrite_row(
            {
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "different"},
                    {"role": "assistant", "content": "answer"},
                ]
            }
        )
