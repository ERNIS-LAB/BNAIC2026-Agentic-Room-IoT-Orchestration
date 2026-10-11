from agent.app import DEFAULT_TOOL_NAMES
from agent.demo import (
    coverage_missing,
    observed_tools_from_final,
    observed_tools_from_payload,
)


def test_demo_extracts_read_sensors_from_trace_payload():
    payload = {
        "name": "tool_call:read_sensors",
        "summary": "",
        "messages": [
            {
                "type": "AIMessage",
                "content": "Calling read_sensors for temperature and light",
            }
        ],
    }

    assert observed_tools_from_payload(payload) == set(DEFAULT_TOOL_NAMES)


def test_demo_extracts_tool_names_from_final_result():
    result = {
        "tool_observations": [
            {
                "tool_name": "read_sensors",
                "input_summary": '{"sensor_names":["temperature"]}',
                "output": "{...}",
            },
        ],
        "run_log": {
            "steps": [
                {
                    "name": "tool_call:read_sensors",
                    "summary": "",
                }
            ]
        },
        "trace": [
            {
                "stage": "tools",
                "payload": {
                    "name": "read_sensors",
                    "summary": "Temperature: 21.5C",
                },
            }
        ],
    }

    assert observed_tools_from_final(result) == set(DEFAULT_TOOL_NAMES)


def test_demo_coverage_reports_missing_tools_in_registry_order():
    observed: set[str] = set()

    assert coverage_missing(observed) == ["read_sensors"]


def test_demo_coverage_passes_when_all_registered_tools_are_seen():
    assert coverage_missing(set(DEFAULT_TOOL_NAMES)) == []
