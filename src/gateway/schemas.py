import time
import uuid
from typing import Any

from pydantic import BaseModel, Field

from agent.schemas import AgentMessage, AgentToolObservation

_created_now = Field(default_factory=lambda: int(time.time()))


class ChatMessage(BaseModel):
    role: str
    content: str | list[Any]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    stop: str | list[str] | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    extra_body: dict[str, Any] | None = None

    model_config = {"extra": "allow"}


class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int = _created_now
    owned_by: str = "local"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelObject]


class ErrorDetail(BaseModel):
    message: str
    type: str
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class AgentQueryRequest(BaseModel):
    thread_id: str = Field(default_factory=lambda: f"thread-{uuid.uuid4().hex[:8]}")
    user_input: str
    prior_messages: list[AgentMessage] = Field(default_factory=list)
    include_trace: bool = False
    enable_thinking: bool = True


class AgentTraceEvent(BaseModel):
    stage: str
    payload: dict[str, Any]


class AgentQueryResponse(BaseModel):
    thread_id: str
    final_answer: str
    tool_observations: list[AgentToolObservation]
    trace: list[AgentTraceEvent] = Field(default_factory=list)
    run_log: dict[str, Any] | None = None
