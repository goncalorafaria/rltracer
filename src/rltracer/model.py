from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

@dataclass(frozen=True)
class Trajectory:
    id: int
    step: int
    prompt_key: str
    messages: list[dict[str, str]]
    metadata: dict[str, Any] = field(default_factory=dict)

class Backend(Protocol):
    def list_steps(self) -> Sequence[int]: ...
    def ensure_step_indexed(self, step: int) -> None: ...
    def list_prompts(self, step: int) -> Sequence[tuple[str, int, str]]: ...
    def list_trajectory_ids(self, step: int, prompt_key: str) -> Sequence[int]: ...
    def load_trajectory(self, trajectory_id: int) -> Trajectory: ...

@dataclass(frozen=True)
class TrajectoryRef:
    id: int
    step: int
    prompt_key: str
    backend: Backend = field(repr=False, compare=False)
    def load(self) -> Trajectory: return self.backend.load_trajectory(self.id)

@dataclass(frozen=True)
class PromptGroup:
    step: int
    key: str
    rollout_count: int
    preview: str
    backend: Backend = field(repr=False, compare=False)
    def trajectories(self, split: str | None = None) -> list[TrajectoryRef]:
        return [TrajectoryRef(i, self.step, self.key, self.backend) for i in self.backend.list_trajectory_ids(self.step, self.key, split)]
    def trajectory(self, index: int = 0) -> TrajectoryRef: return self.trajectories()[index]

@dataclass(frozen=True)
class RolloutStep:
    number: int
    backend: Backend = field(repr=False, compare=False)
    def prompts(self, split: str | None = None) -> list[PromptGroup]:
        self.backend.ensure_step_indexed(self.number)
        return [PromptGroup(self.number, key, count, preview, self.backend) for key, count, preview in self.backend.list_prompts(self.number, split)]

    def prompt(self, key: str, split: str | None = None) -> PromptGroup:
        for group in self.prompts(split):
            if group.key == key:
                return group
        raise KeyError(f"No prompt {key!r} at rollout step {self.number}")

@dataclass(frozen=True)
class TraceRun:
    root: Path
    backend: Backend = field(repr=False, compare=False)
    def steps(self) -> list[RolloutStep]: return [RolloutStep(n, self.backend) for n in self.backend.list_steps()]
    def step(self, number: int) -> RolloutStep: return RolloutStep(number, self.backend)
