import json
from types import SimpleNamespace

import pytest

from agent.iot import StubCameraFrameProvider, StubSensorProvider
from agent.iot.sensors import NUMERIC_SENSOR_NAMES, SENSOR_SPECS
from agent.run_logging import AgentRunJsonlLogger
from agent.schemas import AgentRunLog, AgentStepRecord
from agent.tools import make_read_sensors_tool


def test_stub_provider_requires_fixture_when_not_wired():
    provider = StubCameraFrameProvider()
    with pytest.raises(NotImplementedError):
        provider.fetch_latest_frame()


def test_stub_sensor_provider_returns_deterministic_temperature():
    provider = StubSensorProvider.with_simulated_temperature(value_c=21.5)

    reading = provider.fetch_latest_sensor("temperature")

    assert reading == {
        "sensor": "temperature",
        "value": 21.5,
        "unit": "C",
        "source": "simulated",
    }


def test_read_sensors_tool_returns_structured_readings():
    tool = make_read_sensors_tool(
        StubSensorProvider(
            readings={
                "temperature": {
                    "sensor": "temperature",
                    "value": 21.5,
                    "unit": "C",
                    "source": "simulated",
                }
            }
        )
    )

    result = tool.invoke({"sensor_names": ["temperature"]})

    assert json.loads(result)["sensors"] == ["temperature"]
    assert '"value": 21.5' in result
    assert '"unit": "C"' in result


def test_sensor_specs_are_the_single_sensor_registry():
    assert set(SENSOR_SPECS) == {
        "temperature",
        "temp",
        "light",
        "sound",
        "mic",
        "camera",
    }


def test_read_sensors_keeps_unavailable_sensor_errors():
    tool = make_read_sensors_tool(StubSensorProvider.with_simulated_temperature())
    result = json.loads(tool.invoke({"sensor_names": ["light"]}))
    assert result["readings"]["stub/node_0"]["light"] == {"error": "unavailable"}


def test_camera_tool_preserves_jpeg_content_type():
    tool = make_read_sensors_tool(
        StubSensorProvider.with_simulated_temperature(),
        camera_provider=StubCameraFrameProvider(fixture_bytes=b"\xff\xd8\xff\xd9"),
    )
    result = tool.invoke({"sensor_names": ["camera"]})
    image = next(block for block in result if block["type"] == "image_url")
    assert image["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_read_sensors_tool_schema_restricts_sensor_names():
    tool = make_read_sensors_tool(StubSensorProvider.with_simulated_temperature())

    sensor_enum = tool.args["sensor_names"]["items"]["enum"]

    assert set(sensor_enum) == set(NUMERIC_SENSOR_NAMES) | {"camera"}


def test_run_logger_appends_jsonl_record(tmp_path):
    log_path = tmp_path / "agent-runs.jsonl"
    logger = AgentRunJsonlLogger(log_path)
    run_log = AgentRunLog(
        run_id="run-test",
        thread_id="thread-test",
        user_input="How warm is the room?",
        final_answer="It is 21.5 C.",
        started_at="2026-04-25T10:00:00.000Z",
        ended_at="2026-04-25T10:00:01.000Z",
        total_duration_ms=1000.0,
        tool_observations=[],
        steps=[
            AgentStepRecord(
                sequence=1,
                stage="model",
                pass_number=1,
                name="respond",
                started_at="2026-04-25T10:00:00.000Z",
                ended_at="2026-04-25T10:00:01.000Z",
                duration_ms=1000.0,
                summary="done",
            )
        ],
    )

    logger.append(run_log)

    record = json.loads(log_path.read_text(encoding="utf-8"))
    assert record["run_id"] == "run-test"
    assert record["steps"][0]["duration_ms"] == 1000.0


def test_default_agent_registers_read_sensors_tool(monkeypatch):
    from agent import app as agent_app_module

    captured = {}

    class FakeGraph:
        def stream(self, *args, **kwargs):
            return iter(())

    def fake_build_agent_graph(model, tools, **kwargs):
        captured["model"] = model
        captured["tool_names"] = [tool.name for tool in tools]
        return FakeGraph()

    monkeypatch.setattr(agent_app_module, "build_agent_graph", fake_build_agent_graph)
    settings = SimpleNamespace(
        gateway_base_url="http://localhost:8000",
        llm_backend_v1_url="http://llm:8080/v1",
        model_name="llm",
        backend_url="http://llm:8080",
        request_timeout=120.0,
        camera_frame_fixture_path=None,
        simulated_temperature_c=21.5,
        agent_max_iterations=4,
        agent_max_context_messages=10,
        agent_enable_memory=False,
        agent_run_log_path=None,
        langfuse_enabled=False,
        langfuse_environment="test",
        langfuse_trace_name="mrp-agent-run",
        langfuse_flush_on_run_end=False,
        default_temperature=1.0,
        default_top_p=0.95,
        default_presence_penalty=0.0,
        default_top_k=20,
    )

    agent_app_module.build_agent_app(settings=settings)

    assert captured["tool_names"] == ["read_sensors"]
    assert captured["model"].streaming is True


def test_default_tool_registry_names_match_demo_coverage_list():
    from agent.app import DEFAULT_TOOL_NAMES, build_tools

    settings = SimpleNamespace(
        camera_frame_fixture_path=None,
        simulated_temperature_c=21.5,
    )

    tools = build_tools(
        settings=settings,
        camera_provider=StubCameraFrameProvider(fixture_bytes=b"png-bytes"),
        sensor_provider=StubSensorProvider(
            readings={
                "temperature": {
                    "sensor": "temperature",
                    "value": 21.5,
                    "unit": "C",
                    "source": "simulated",
                }
            }
        ),
    )

    assert tuple(tool.name for tool in tools) == DEFAULT_TOOL_NAMES
