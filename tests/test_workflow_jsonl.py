import json

from rltracer.workflow_jsonl import WorkflowJSONLAdapter


def test_workflow_jsonl_adapter_reads_intermediate_messages(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps({"route": "submit", "final_row": {"rubric": {"messages": json.dumps([
        {"role": "system", "content": "judge"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "checking", "tool_calls": [{"name": "terminal"}]},
    ])}}}) + "\n")
    adapter = WorkflowJSONLAdapter(path)
    try:
        assert adapter.list_steps() == [0]
        prompts = adapter.list_prompts(0)
        assert len(prompts) == 1
        trajectory = adapter.load_trajectory(adapter.list_trajectory_ids(0, prompts[0][0])[0])
        assert [message["role"] for message in trajectory.messages] == ["system", "user", "assistant"]
        assert trajectory.messages[-1]["tool_calls"] == [{"name": "terminal"}]
    finally:
        adapter.close()
