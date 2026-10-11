from __future__ import annotations

import pytest
from langchain.tools import tool
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from agent.app import AgentApp
from agent.schemas import AgentMessage, AgentRequest


class FakeModel(FakeMessagesListChatModel):
    """FakeMessagesListChatModel extended to support bind_tools (returns self unchanged)."""

    def bind_tools(self, tools, **kwargs):
        return self


@tool
def get_current_camera_frame() -> list[dict]:
    """Return the current camera frame."""
    return [
        {"type": "text", "text": "Current camera frame."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,cG5n"}},
    ]


@tool
def get_current_room_context(prompt: str) -> str:
    """Return known room metadata."""
    return "Room Atlas seats 6 people."


@tool
def get_current_temperature() -> str:
    """Return current room temperature."""
    return (
        '{"sensor": "temperature", "value": 21.5, "unit": "C", "source": "simulated"}'
    )


def _make_tool_call_message(
    tool_name: str, args: dict, call_id: str = "call_1"
) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": tool_name, "args": args, "id": call_id, "type": "tool_call"}
        ],
    )


def test_direct_response_path_completes_without_tool_use():
    model = FakeModel(
        responses=[AIMessage(content="APIs let systems talk to each other.")]
    )
    app = AgentApp(model=model, tools=[])

    result = app.invoke(
        AgentRequest(thread_id="thread-1", user_input="What is an API?")
    )

    assert result.final_answer == "APIs let systems talk to each other."
    assert result.tool_observations == []


def test_tool_path_collects_observation_then_finishes():
    model = FakeModel(
        responses=[
            _make_tool_call_message("get_current_camera_frame", {}, "call_1"),
            AIMessage(content="The room is empty."),
        ]
    )
    app = AgentApp(model=model, tools=[get_current_camera_frame])

    result = app.invoke(
        AgentRequest(thread_id="thread-2", user_input="Is anyone in the room?")
    )

    assert result.final_answer == "The room is empty."
    assert len(result.tool_observations) == 1
    assert result.tool_observations[0].tool_name == "get_current_camera_frame"
    assert result.tool_observations[0].output[1]["type"] == "image_url"


def test_multi_tool_path_uses_camera_frame_then_room_context():
    model = FakeModel(
        responses=[
            _make_tool_call_message("get_current_camera_frame", {}, "call_1"),
            _make_tool_call_message(
                "get_current_room_context", {"prompt": "visitor briefing"}, "call_2"
            ),
            AIMessage(
                content="Two people in Room Atlas, a sixth-floor hybrid room for up to 6."
            ),
        ]
    )
    app = AgentApp(
        model=model, tools=[get_current_camera_frame, get_current_room_context]
    )

    result = app.invoke(
        AgentRequest(thread_id="thread-multi", user_input="Brief me on the room.")
    )

    tool_names = [obs.tool_name for obs in result.tool_observations]
    assert tool_names == ["get_current_camera_frame", "get_current_room_context"]
    assert "Two people" in result.final_answer


def test_trace_records_model_and_tool_stages():
    model = FakeModel(
        responses=[
            _make_tool_call_message("get_current_camera_frame", {}, "call_1"),
            AIMessage(content="Done."),
        ]
    )
    app = AgentApp(model=model, tools=[get_current_camera_frame])

    result = app.invoke(
        AgentRequest(thread_id="thread-trace", user_input="What is happening?")
    )

    stages = [event["stage"] for event in result.trace]
    assert "model" in stages
    assert "tools" in stages


def test_prior_assistant_messages_are_not_reported_as_current_trace():
    model = FakeModel(
        responses=[AIMessage(content="This answer came from the current graph run.")]
    )
    app = AgentApp(model=model, tools=[])

    result = app.invoke(
        AgentRequest(
            thread_id="thread-history",
            user_input="What is happening now?",
            prior_messages=[
                AgentMessage(role="user", content="What happened before?"),
                AgentMessage(role="assistant", content="This was an earlier answer."),
            ],
        )
    )

    assert result.final_answer == "This answer came from the current graph run."
    assert len(result.trace) == 1
    assert result.trace[0]["stage"] == "model"
    assert result.trace[0]["payload"]["duration_ms"] >= 0
    assert (
        result.trace[0]["payload"]["summary"]
        == "This answer came from the current graph run."
    )


def test_iter_events_yields_trace_and_final_result():
    model = FakeModel(
        responses=[
            _make_tool_call_message("get_current_camera_frame", {}, "call_1"),
            AIMessage(content="The room is empty."),
        ]
    )
    app = AgentApp(model=model, tools=[get_current_camera_frame])

    events = list(
        app.iter_events(
            AgentRequest(thread_id="thread-events", user_input="What is happening?")
        )
    )

    event_types = [e["event"] for e in events]
    assert "trace" in event_types
    assert event_types[-1] == "final"
    assert events[-1]["result"].final_answer == "The room is empty."
    assert events[-1]["result"].run_log is not None
    assert events[-1]["result"].run_log.steps[0].duration_ms >= 0


def test_temperature_tool_path_collects_deterministic_observation():
    model = FakeModel(
        responses=[
            _make_tool_call_message("get_current_temperature", {}, "call_1"),
            AIMessage(content="The room is 21.5 C."),
        ]
    )
    app = AgentApp(model=model, tools=[get_current_temperature])

    result = app.invoke(
        AgentRequest(thread_id="thread-temp", user_input="How warm is it?")
    )

    assert result.final_answer == "The room is 21.5 C."
    assert result.tool_observations[0].tool_name == "get_current_temperature"
    assert result.tool_observations[0].output["value"] == 21.5


def test_camera_and_temperature_trace_records_ordered_timing_steps():
    model = FakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_current_camera_frame",
                        "args": {},
                        "id": "call_1",
                        "type": "tool_call",
                    },
                    {
                        "name": "get_current_temperature",
                        "args": {},
                        "id": "call_2",
                        "type": "tool_call",
                    },
                ],
            ),
            AIMessage(content="The room is empty and 21.5 C."),
        ]
    )
    app = AgentApp(
        model=model,
        tools=[get_current_camera_frame, get_current_temperature],
    )

    result = app.invoke(
        AgentRequest(thread_id="thread-camera-temp", user_input="Check room status.")
    )

    assert result.run_log is not None
    assert [step.sequence for step in result.run_log.steps] == list(
        range(1, len(result.run_log.steps) + 1)
    )
    assert all(step.duration_ms >= 0 for step in result.run_log.steps)
    tool_steps = [step for step in result.run_log.steps if step.stage == "tools"]
    assert tool_steps
    assert any("get_current_camera_frame" in step.name for step in tool_steps)
    assert any("get_current_temperature" in step.name for step in tool_steps)


def test_iter_events_ignores_empty_update_payloads():
    model = FakeModel(responses=[AIMessage(content="Done.")])
    app = AgentApp(model=model, tools=[])

    def fake_stream(*args, **kwargs):
        yield "updates", {"model": None}
        yield "updates", {"model": {"messages": [AIMessage(content="Done.")]}}

    app._graph.stream = fake_stream

    events = list(
        app.iter_events(AgentRequest(thread_id="thread-empty-update", user_input="Hi."))
    )

    assert events[-1]["event"] == "final"
    assert events[-1]["result"].final_answer == "Done."


def test_langfuse_disabled_does_not_change_graph_config():
    app = AgentApp(model=FakeModel(responses=[AIMessage(content="Done.")]), tools=[])

    config = app._config("thread-config", "run-config")

    assert "callbacks" not in config
    assert config["metadata"] == {
        "thread_id": "thread-config",
        "agent_run_id": "run-config",
    }


def test_langfuse_tracing_config_is_merged_into_graph_config():
    app = AgentApp(model=FakeModel(responses=[AIMessage(content="Done.")]), tools=[])

    config = app._config(
        "thread-config",
        "run-config",
        {
            "callbacks": ["callback"],
            "run_name": "mrp-agent-run",
            "tags": ["mrp-agent", "test"],
            "metadata": {"langfuse_session_id": "thread-config"},
        },
    )

    assert config["callbacks"] == ["callback"]
    assert config["run_name"] == "mrp-agent-run"
    assert config["tags"] == ["mrp-agent", "test"]
    assert config["metadata"]["thread_id"] == "thread-config"
    assert config["metadata"]["agent_run_id"] == "run-config"
    assert config["metadata"]["langfuse_session_id"] == "thread-config"


def test_recursion_limit_raises_when_agent_loops():
    model = FakeModel(
        responses=[
            _make_tool_call_message("get_current_camera_frame", {}, "call_1"),
            _make_tool_call_message("get_current_camera_frame", {}, "call_2"),
            _make_tool_call_message("get_current_camera_frame", {}, "call_3"),
        ]
    )
    app = AgentApp(model=model, tools=[get_current_camera_frame], max_iterations=1)

    with pytest.raises(Exception):
        app.invoke(AgentRequest(thread_id="thread-limit", user_input="Keep looking."))
