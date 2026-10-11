"""Typed request and response models for the agent framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel


class AgentMessage(BaseModel):
    role: str
    content: str


class AgentToolObservation(BaseModel):
    tool_name: str
    input_summary: str
    output: Any  # string (sensor JSON) or list (camera content blocks)


@dataclass
class AgentStepRecord:
    sequence: int
    stage: str
    pass_number: int
    name: str
    started_at: str
    ended_at: str
    duration_ms: float
    usage: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    tool_timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "stage": self.stage,
            "pass_number": self.pass_number,
            "name": self.name,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "usage": self.usage,
            "timings": self.timings,
            "summary": self.summary,
            "tool_timings": self.tool_timings,
        }


@dataclass
class AgentRunLog:
    run_id: str
    thread_id: str
    user_input: str
    final_answer: str
    started_at: str
    ended_at: str
    total_duration_ms: float
    tool_observations: list[AgentToolObservation]
    steps: list[AgentStepRecord]
    cost_summary: dict[str, Any] = field(default_factory=dict)
    structured_response: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "run_id": self.run_id,
            "thread_id": self.thread_id,
            "user_input": self.user_input,
            "final_answer": self.final_answer,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "total_duration_ms": self.total_duration_ms,
            "cost_summary": self.cost_summary,
            "tool_observations": [obs.model_dump() for obs in self.tool_observations],
            "steps": [step.to_dict() for step in self.steps],
        }
        if self.structured_response is not None:
            d["structured_response"] = self.structured_response
        return d


@dataclass
class AgentRequest:
    thread_id: str
    user_input: str
    prior_messages: list[AgentMessage] = field(default_factory=list)
    enable_thinking: bool = True


@dataclass
class GraphResult:
    thread_id: str
    final_answer: str
    tool_observations: list[AgentToolObservation]
    trace: list[dict[str, Any]] = field(default_factory=list)
    run_log: AgentRunLog | None = None
    structured_response: dict[str, Any] | None = None
