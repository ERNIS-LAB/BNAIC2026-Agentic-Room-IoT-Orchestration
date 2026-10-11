"""Application assembly for the MRP agent."""

import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import SecretStr

from agent.clients import ChatOpenAIWithTimings
from agent.graph.workflow import ResponseFormatArg, build_agent_graph
from agent.iot import (
    CameraFrameProvider,
    SensorProvider,
    StubCameraFrameProvider,
    StubSensorProvider,
)
from agent.observability import LangfuseConfig, LangfuseTracer
from agent.run_logging import AgentRunJsonlLogger
from agent.schemas import (
    AgentRequest,
    AgentRunLog,
    AgentStepRecord,
    AgentToolObservation,
    GraphResult,
)
from agent.tools import make_read_sensors_tool

DEFAULT_TOOL_NAMES = ("read_sensors",)


@dataclass
class AgentRuntimeContext:
    thread_id: str


def _build_messages(
    request: AgentRequest, *, include_prior_messages: bool = True
) -> list:
    messages = []
    if include_prior_messages:
        for msg in request.prior_messages:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            else:
                messages.append(AIMessage(content=msg.content))
    messages.append(HumanMessage(content=request.user_input))
    return messages


def _latest_ai_message(messages: list) -> AIMessage | None:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg
    return None


def _token_text(token) -> str:
    text = getattr(token, "text", "")
    if isinstance(text, str) and text:
        return text
    content = getattr(token, "content", "")
    return content if isinstance(content, str) else ""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _message_summary(messages: list) -> str:
    if not messages:
        return ""
    msg = messages[-1]
    content = getattr(msg, "content", "")
    if isinstance(msg, AIMessage) and msg.tool_calls:
        return f"tool_call: {_tool_call_names(msg)}"
    if isinstance(msg, ToolMessage):
        return f"{msg.name}: {_content_summary(content)}"
    return _content_summary(content)


def _parse_tool_content(content) -> Any:
    """Return structured tool output for the frontend.

    - Camera frames (list[dict]): returned as-is so the frontend can render the image.
    - Sensor readings (JSON string): parsed into a dict.
    - Other: kept as a string.
    """
    if isinstance(content, list):
        return content
    if isinstance(content, str) and content.strip().startswith("{"):
        try:
            return json.loads(content)
        except (json.JSONDecodeError, ValueError):
            pass
    return str(content)


def _content_summary(content) -> str:
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block[:80])
            elif isinstance(block, dict):
                block_type = block.get("type", "dict")
                if block_type == "image_url":
                    parts.append("[image_url]")
                elif block_type == "text":
                    parts.append(str(block.get("text", ""))[:80])
                else:
                    parts.append(f"[{block_type}]")
        return " ".join(part for part in parts if part)[:160]
    return str(content)[:160]


def _tool_call_names(message: AIMessage) -> str:
    return ", ".join(str(tc.get("name", "")) for tc in message.tool_calls)


def _step_name(stage: str, messages: list) -> str:
    if stage == "model":
        ai_msg = _latest_ai_message(messages)
        if ai_msg is not None and ai_msg.tool_calls:
            return f"tool_call:{_tool_call_names(ai_msg)}"
        return "respond"
    if stage == "tools":
        names = [
            str(getattr(msg, "name", ""))
            for msg in messages
            if isinstance(msg, ToolMessage)
        ]
        return ", ".join(name for name in names if name) or "tools"
    return stage


def _result_from_messages(
    thread_id: str,
    messages: list,
    structured_response: dict[str, Any] | None = None,
) -> GraphResult:
    final_answer = ""
    tool_observations: list[AgentToolObservation] = []
    tool_call_inputs: dict[str, str] = {}

    # If the graph produced a structured response via ToolStrategy, extract
    # the "answer" field as the canonical final_answer.
    if structured_response:
        final_answer = str(structured_response.get("answer", ""))
        # Fallback: if the answer field is empty, try the free-text messages.
        if not final_answer.strip():
            structured_response = None

    for msg in messages:
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    if structured_response is not None and tc.get("name") in (
                        "AgentResponse",
                        "respond",
                    ):
                        continue
                    tool_call_inputs[str(tc.get("id"))] = json.dumps(tc.get("args", {}))
            elif structured_response is None and msg.content:
                final_answer = str(msg.content)
        elif isinstance(msg, ToolMessage):
            tool_observations.append(
                AgentToolObservation(
                    tool_name=msg.name or "",
                    input_summary=tool_call_inputs.get(msg.tool_call_id, ""),
                    output=_parse_tool_content(msg.content),
                )
            )

    return GraphResult(
        thread_id=thread_id,
        final_answer=final_answer,
        tool_observations=tool_observations,
        structured_response=structured_response,
    )


def _extract_llm_ms(timings: dict) -> tuple[float, float]:
    """Return (prompt_eval_ms, eval_ms) from a llama.cpp timings dict."""
    if "prompt_ms" in timings:
        return timings.get("prompt_ms", 0.0), timings.get("predicted_ms", 0.0)
    # Older llama.cpp format uses nanoseconds
    return (
        timings.get("prompt_eval_duration", 0) / 1e6,
        timings.get("eval_duration", 0) / 1e6,
    )


def _compute_cost_summary(steps: list) -> dict:
    model_ms = tools_ms = overhead_ms = 0.0
    prompt_tokens = completion_tokens = 0
    llm_prompt_eval_ms = llm_eval_ms = 0.0
    tool_breakdown: dict[str, float] = {}

    for step in steps:
        if step.stage == "model":
            model_ms += step.duration_ms
            prompt_tokens += step.usage.get(
                "input_tokens", step.usage.get("prompt_tokens", 0)
            )
            completion_tokens += step.usage.get(
                "output_tokens", step.usage.get("completion_tokens", 0)
            )
            pe_ms, e_ms = _extract_llm_ms(step.timings)
            llm_prompt_eval_ms += pe_ms
            llm_eval_ms += e_ms
        elif step.stage == "tools":
            tools_ms += step.duration_ms
            for name, dur in step.tool_timings.items():
                tool_breakdown[name] = round(tool_breakdown.get(name, 0.0) + dur, 3)
        else:
            overhead_ms += step.duration_ms

    return {
        "model_total_ms": round(model_ms, 3),
        "tools_total_ms": round(tools_ms, 3),
        "overhead_ms": round(overhead_ms, 3),
        "total_prompt_tokens": prompt_tokens,
        "total_completion_tokens": completion_tokens,
        "llm_prompt_eval_ms": round(llm_prompt_eval_ms, 3),
        "llm_eval_ms": round(llm_eval_ms, 3),
        "tool_breakdown": tool_breakdown,
    }


class AgentApp:
    """Stateless entrypoint that wraps the compiled LangChain agent graph."""

    def __init__(
        self,
        model,
        tools: list,
        max_iterations: int = 4,
        max_context_messages: int = 10,
        enable_memory: bool = True,
        run_log_path: str | Path | None = None,
        model_name: str = "llm",
        observability: LangfuseTracer | None = None,
        sensor_provider: SensorProvider | None = None,
        model_nonthinking=None,  # second model for non-thinking requests
        response_format: ResponseFormatArg = None,
    ) -> None:
        self._max_iterations = max_iterations
        self._model_name = model_name
        self._checkpointer = InMemorySaver() if enable_memory else None
        self._seeded_threads: set[str] = set()
        self._seeded_threads_lock = Lock()
        self._graph = build_agent_graph(
            model,
            tools,
            checkpointer=self._checkpointer,
            context_schema=AgentRuntimeContext,
            max_context_messages=max_context_messages,
            response_format=response_format,
        )
        # Build a second graph for non-thinking requests, sharing the same
        # checkpointer so conversation history is visible to both.
        if model_nonthinking is not None:
            self._graph_nonthinking = build_agent_graph(
                model_nonthinking,
                tools,
                checkpointer=self._checkpointer,
                context_schema=AgentRuntimeContext,
                max_context_messages=max_context_messages,
                response_format=response_format,
            )
        else:
            self._graph_nonthinking = None
        self._run_logger = AgentRunJsonlLogger(run_log_path)
        self._observability = observability or LangfuseTracer()
        self._sensor_provider = sensor_provider

    def _config(
        self,
        thread_id: str,
        run_id: str,
        tracing_config: dict[str, Any] | None = None,
    ) -> dict:
        config = {
            "recursion_limit": self._max_iterations * 4 + 3,
            "max_concurrency": 4,
            "configurable": {"thread_id": thread_id},
            "metadata": {"thread_id": thread_id, "agent_run_id": run_id},
        }
        resolved_tracing_config = (
            tracing_config
            or self._observability.langchain_config(
                run_id=run_id,
                thread_id=thread_id,
            )
        )
        if resolved_tracing_config:
            config["callbacks"] = resolved_tracing_config.get("callbacks", [])
            config["run_name"] = resolved_tracing_config.get("run_name")
            config["tags"] = resolved_tracing_config.get("tags", [])
            config["metadata"].update(resolved_tracing_config.get("metadata", {}))
        return config

    def _should_seed_thread(self, request: AgentRequest) -> bool:
        if self._checkpointer is None or not request.prior_messages:
            return bool(request.prior_messages)
        with self._seeded_threads_lock:
            if request.thread_id in self._seeded_threads:
                return False
            self._seeded_threads.add(request.thread_id)
            return True

    def invoke(self, request: AgentRequest) -> GraphResult:
        for event in self.iter_events(request):
            if event.get("event") == "final":
                return event["result"]
        return GraphResult(
            thread_id=request.thread_id,
            final_answer="",
            tool_observations=[],
            trace=[],
        )

    def iter_events(self, request: AgentRequest):
        run_id = f"run-{uuid.uuid4().hex[:12]}"
        run_started_at = _utc_now()
        run_started_perf = step_started_perf = time.perf_counter()
        step_started_at = run_started_at
        messages = _build_messages(
            request,
            include_prior_messages=self._should_seed_thread(request),
        )
        # Inject known IoT nodes into the user message so the agent always
        # knows which nodes are online without needing a discovery tool.
        if self._sensor_provider is not None:
            try:
                known = self._sensor_provider.list_nodes()
            except Exception:
                known = []
            if known:
                node_list = ", ".join(known)
                last = messages[-1]
                messages[-1] = HumanMessage(
                    content=(
                        f"[System context: these IoT nodes are currently online: "
                        f"{node_list}]\n\n{last.content}"
                    )
                )
        all_messages: list = []
        trace: list[dict[str, Any]] = []
        steps: list[AgentStepRecord] = []
        stage_counts: dict[str, int] = {}
        structured_response: dict[str, Any] | None = None
        step_tool_timings: dict[str, float] = {}
        # Select the graph whose model has the right thinking mode baked in.
        if request.enable_thinking or self._graph_nonthinking is None:
            graph = self._graph
        else:
            graph = self._graph_nonthinking
        config = self._config(request.thread_id, run_id)
        stream = graph.stream(
            {"messages": messages},
            config=config,
            context=AgentRuntimeContext(thread_id=request.thread_id),
            stream_mode=["messages", "updates"],
        )
        for stream_mode, stream_data in stream:
            if stream_mode == "messages":
                token, metadata = stream_data
                if str(metadata.get("langgraph_node", "")) == "model":
                    reasoning = token.additional_kwargs.get("reasoning_content")
                    if reasoning:
                        yield {
                            "event": "reasoning_token",
                            "stage": "model",
                            "payload": {"text": reasoning},
                        }
                    text = _token_text(token)
                    if text:
                        yield {
                            "event": "token",
                            "stage": "model",
                            "payload": {"text": text},
                        }
                continue

            if stream_mode != "updates":
                continue

            for node_name, node_output in stream_data.items():
                if not isinstance(node_output, dict):
                    continue
                node_name = str(node_name)

                # Capture structured_response from ToolStrategy output.
                sr = node_output.get("structured_response")
                if sr is not None:
                    if isinstance(sr, dict):
                        structured_response = sr
                    else:
                        # Pydantic model — serialise to dict.
                        try:
                            structured_response = sr.model_dump()
                        except AttributeError:
                            structured_response = {"answer": str(sr)}

                now = _utc_now()
                now_perf = time.perf_counter()
                node_messages = node_output.get("messages", [])
                all_messages.extend(node_messages)
                stage_counts[node_name] = stage_counts.get(node_name, 0) + 1
                payload: dict[str, Any] = {
                    "messages": [
                        {
                            "type": type(m).__name__,
                            "content": _content_summary(m.content),
                        }
                        for m in node_messages
                    ]
                }
                if node_name == "model":
                    ai_msg = _latest_ai_message(node_messages)
                    if ai_msg is not None:
                        payload["model"] = self._model_name
                        payload["usage"] = dict(ai_msg.usage_metadata or {})
                        payload["timings"] = ai_msg.additional_kwargs.get("timings", {})
                duration_ms = round((now_perf - step_started_perf) * 1000, 3)
                step = AgentStepRecord(
                    sequence=len(steps) + 1,
                    stage=node_name,
                    pass_number=stage_counts[node_name],
                    name=_step_name(node_name, node_messages),
                    started_at=_iso(step_started_at),
                    ended_at=_iso(now),
                    duration_ms=duration_ms,
                    usage=dict(payload.get("usage") or {}),
                    timings=dict(payload.get("timings") or {}),
                    summary=_message_summary(node_messages),
                    tool_timings=step_tool_timings,
                )
                steps.append(step)
                payload.update(
                    {
                        "sequence": step.sequence,
                        "pass_number": step.pass_number,
                        "name": step.name,
                        "started_at": step.started_at,
                        "ended_at": step.ended_at,
                        "duration_ms": step.duration_ms,
                        "summary": step.summary,
                    }
                )
                trace.append({"stage": node_name, "payload": payload})
                yield {
                    "event": "trace",
                    "stage": node_name,
                    "payload": payload,
                }
                step_started_at = now
                step_started_perf = now_perf
        run_ended_at = _utc_now()
        result = _result_from_messages(
            request.thread_id,
            all_messages,
            structured_response=structured_response,
        )
        result.trace = trace
        run_log = AgentRunLog(
            run_id=run_id,
            thread_id=request.thread_id,
            user_input=request.user_input,
            final_answer=result.final_answer,
            started_at=_iso(run_started_at),
            ended_at=_iso(run_ended_at),
            total_duration_ms=round((time.perf_counter() - run_started_perf) * 1000, 3),
            tool_observations=result.tool_observations,
            steps=steps,
            cost_summary=_compute_cost_summary(steps),
            structured_response=structured_response,
        )
        result.run_log = run_log
        self._run_logger.append(run_log)
        self._observability.flush()
        yield {"event": "final", "result": result}


def build_tools(
    settings=None,
    camera_provider: CameraFrameProvider | None = None,
    sensor_provider: SensorProvider | None = None,
) -> list:
    from gateway.config import settings as default_settings

    resolved = settings or default_settings
    camera = camera_provider or (
        StubCameraFrameProvider(fixture_path=Path(resolved.camera_frame_fixture_path))
        if resolved.camera_frame_fixture_path
        else StubCameraFrameProvider()
    )
    sensors = sensor_provider or StubSensorProvider.with_simulated_temperature(
        value_c=resolved.simulated_temperature_c
    )
    return [
        make_read_sensors_tool(sensors, camera_provider=camera),
    ]


def build_agent_app(
    settings=None,
    camera_provider: CameraFrameProvider | None = None,
    sensor_provider: SensorProvider | None = None,
    *,
    llm_base_url: str | None = None,
) -> AgentApp:
    from gateway.config import settings as default_settings

    resolved = settings or default_settings
    base = llm_base_url or f"{resolved.gateway_base_url.rstrip('/')}/v1"
    llm_timeout = resolved.request_timeout

    model_kwargs = dict(
        base_url=base,
        model=resolved.model_name,
        timeout=llm_timeout,
        max_retries=0,
        streaming=True,
        api_key=SecretStr("not-needed"),
        temperature=resolved.default_temperature,
        top_p=resolved.default_top_p,
        presence_penalty=resolved.default_presence_penalty,
    )
    model_thinking = ChatOpenAIWithTimings(
        **model_kwargs,
        extra_body={
            "top_k": resolved.default_top_k,
            "chat_template_kwargs": {
                "enable_thinking": True,
                "preserve_thinking": True,
            },
        },
    )
    model_nonthinking = ChatOpenAIWithTimings(
        **model_kwargs,
        extra_body={
            "top_k": resolved.default_top_k,
            "chat_template_kwargs": {
                "enable_thinking": False,
                "preserve_thinking": False,
            },
        },
    )
    # Build structured-output strategy.  ToolStrategy works with any backend
    # that supports tool calling (including llama.cpp/Qwen).  handle_errors=True
    # tells LangChain to retry once on a parse failure instead of crashing.
    from langchain.agents.structured_output import ToolStrategy

    from agent.response_schema import AgentResponse

    structured_output = ToolStrategy(schema=AgentResponse, handle_errors=True)

    return AgentApp(
        model=model_thinking,
        tools=build_tools(
            settings=resolved,
            camera_provider=camera_provider,
            sensor_provider=sensor_provider,
        ),
        max_iterations=resolved.agent_max_iterations,
        max_context_messages=resolved.agent_max_context_messages,
        enable_memory=resolved.agent_enable_memory,
        run_log_path=resolved.agent_run_log_path,
        model_name=resolved.model_name,
        observability=LangfuseTracer(
            LangfuseConfig(
                enabled=resolved.langfuse_enabled,
                environment=resolved.langfuse_environment,
                trace_name=resolved.langfuse_trace_name,
                flush_on_run_end=resolved.langfuse_flush_on_run_end,
            )
        ),
        sensor_provider=sensor_provider,
        model_nonthinking=model_nonthinking,
        response_format=structured_output,
    )
