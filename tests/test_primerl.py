import msgpack
from pathlib import Path
from rltracer.primerl import PrimeRLAdapter

class Tokenizer:
    def decode(self, ids, skip_special_tokens=False): return "".join(map(chr, ids))

def tokens(text): return list(map(ord, text))
def chat(user, answer): return f"<|im_start|>system\ns<|im_end|><|im_start|>user\n{user}<|im_end|><|im_start|>assistant\n{answer}<|im_end|>"

def test_index_and_lazy_load(tmp_path: Path):
    shard = tmp_path / "rollouts/step_1/rank_0.bin"; shard.parent.mkdir(parents=True)
    shard.write_bytes(msgpack.packb([[tokens(chat("one","a") + chat("two","b"))]]))
    adapter = PrimeRLAdapter(tmp_path / "rollouts", "unused"); adapter.tokenizer = Tokenizer()
    adapter.ensure_step_indexed(1)
    prompts = adapter.list_prompts(1)
    assert len(prompts) == 2
    trajectory = adapter.load_trajectory(adapter.list_trajectory_ids(1, prompts[0][0])[0])
    assert [m["role"] for m in trajectory.messages] == ["system", "user", "assistant"]
