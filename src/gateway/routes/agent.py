import json
import logging

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from agent.schemas import AgentMessage, AgentRequest
from gateway.schemas import (
    AgentQueryRequest,
    AgentQueryResponse,
    AgentTraceEvent,
)
from gateway.services.agent_runtime import get_agent_app

router = APIRouter()
logger = logging.getLogger(__name__)


def _build_agent_request(request: AgentQueryRequest) -> AgentRequest:
    return AgentRequest(
        thread_id=request.thread_id,
        user_input=request.user_input,
        prior_messages=[
            AgentMessage(role=m.role, content=m.content) for m in request.prior_messages
        ],
        enable_thinking=request.enable_thinking,
    )


def _build_agent_response(result, *, include_trace: bool) -> AgentQueryResponse:
    return AgentQueryResponse(
        thread_id=result.thread_id,
        final_answer=result.final_answer,
        tool_observations=list(result.tool_observations),
        trace=[
            AgentTraceEvent(
                stage=str(event.get("stage", "")),
                payload=_safe_payload(event),
            )
            for event in result.trace
        ]
        if include_trace
        else [],
        run_log=result.run_log.to_dict() if getattr(result, "run_log", None) else None,
    )


def _ndjson_line(payload: dict) -> bytes:
    return (json.dumps(payload) + "\n").encode("utf-8")


def _safe_payload(event: dict) -> dict:
    p = event.get("payload")
    return p if isinstance(p, dict) else {}


_STREAM_THROUGH_EVENTS = frozenset({"reasoning_token", "trace", "token"})


@router.post(
    "/v1/agent/query",
    response_model=AgentQueryResponse,
    description="Endpoint to query the agent with a user input and receive a final answer along with optional trace information.",
)
async def agent_query(request: AgentQueryRequest) -> AgentQueryResponse:
    agent_app = get_agent_app()
    agent_request = _build_agent_request(request)
    result = await run_in_threadpool(agent_app.invoke, agent_request)
    return _build_agent_response(result, include_trace=request.include_trace)


@router.post(
    "/v1/agent/stream",
    description="Endpoint to query the agent and receive a stream of events including token generation, trace updates, and the final answer.",
)
async def agent_stream(request: AgentQueryRequest) -> StreamingResponse:
    agent_app = get_agent_app()
    agent_request = _build_agent_request(request)

    def event_stream():
        try:
            for event in agent_app.iter_events(agent_request):
                event_type = event.get("event")
                if event_type in _STREAM_THROUGH_EVENTS:
                    yield _ndjson_line(
                        {
                            "event": event_type,
                            "stage": str(event.get("stage", "")),
                            "payload": _safe_payload(event),
                        }
                    )
                elif event_type == "final":
                    response = _build_agent_response(
                        event.get("result"),
                        include_trace=True,
                    )
                    yield _ndjson_line(
                        {
                            "event": "final",
                            "result": response.model_dump(),
                        }
                    )
        except Exception as exc:
            logger.exception("Agent demo stream failed")
            yield _ndjson_line(
                {
                    "event": "error",
                    "detail": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                }
            )

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
    )
