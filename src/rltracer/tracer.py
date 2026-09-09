"""Top-level facade for lazily exploring one RL trace run."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .model import RolloutStep


class RunAdapter(Protocol):
    rollout_dir: Path

    def list_steps(self) -> list[int]: ...


class RLTracer:
    """Explore one RL run through a configured library-specific adapter."""

    def __init__(self, adapter: RunAdapter):
        self.adapter = adapter
        self.root = adapter.rollout_dir.parent

    def steps(self) -> list[RolloutStep]:
        return [RolloutStep(number, self.adapter) for number in self.adapter.list_steps()]

    def step(self, number: int) -> RolloutStep:
        if number not in self.adapter.list_steps():
            raise KeyError(f"No rollout step {number} under {self.root}")
        return RolloutStep(number, self.adapter)

    def close(self) -> None:
        close = getattr(self.adapter, "close", None)
        if close:
            close()
