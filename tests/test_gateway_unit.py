import json

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from gateway.config import settings
from gateway.main import app
from gateway.routes import agent as agent_routes
from gateway.routes.chat import _get_backend_url, _with_generation_defaults
from gateway.schemas import ChatCompletionRequest


def test_get_backend_url_accepts_configured_model():
    assert _get_backend_url(settings.model_name) == settings.backend_url


def test_get_backend_url_rejects_unknown_model():
    with pytest.raises(HTTPException) as exc_info:
        _get_backend_url("unknown-model")

    assert exc_info.value.status_code == 404


def test_generation_defaults_are_added_when_missing():
    payload = _with_generation_defaults(
        {"model": "llm", "messages": [{"role": "user", "content": "hello"}]}
    )

    assert payload["temperature"] == settings.default_temperature
    assert payload["top_p"] == settings.default_top_p
    assert payload["presence_penalty"] == settings.default_presence_penalty
    assert payload["top_k"] == settings.default_top_k


def test_generation_defaults_do_not_override_callers():
    payload = _with_generation_defaults(
        {
            "model": "llm",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.2,
            "top_p": 0.8,
            "presence_penalty": 0.0,
            "top_k": 10,
        }
    )

    assert payload["temperature"] == 0.2
    assert payload["top_p"] == 0.8
    assert payload["presence_penalty"] == 0.0
    assert payload["top_k"] == 10


def test_extra_body_merges_without_overriding_top_level_values():
    payload = _with_generation_defaults(
        {
            "model": "llm",
            "messages": [{"role": "user", "content": "hello"}],
            "top_k": 10,
            "extra_body": {
                "top_k": 20,
                "chat_template_kwargs": {"preserve_thinking": True},
            },
        }
    )

    assert "extra_body" not in payload
    assert payload["top_k"] == 10
    assert payload["chat_template_kwargs"] == {"preserve_thinking": True}


def test_chat_request_preserves_openai_sdk_extra_parameters():
    request = ChatCompletionRequest(
        model="llm",
        messages=[{"role": "user", "content": "hello"}],
        chat_template_kwargs={"preserve_thinking": True},
    )

    payload = request.model_dump(exclude_none=True)
    assert payload["chat_template_kwargs"] == {"preserve_thinking": True}


def test_agent_stream_returns_ndjson_events(monkeypatch):
    class FakeRunLog:
        def to_dict(self):
            return {"thread_id": "stream-test", "steps": []}

    class FakeResult:
        thread_id = "stream-test"
        final_answer = "Done."
        tool_observations = []
        trace = []
        run_log = FakeRunLog()

    class FakeAgentApp:
        def iter_events(self, request):
            assert request.thread_id == "stream-test"
            assert request.user_input == "hello"
            yield {"event": "token", "stage": "model", "payload": {"text": "Do"}}
            yield {"event": "trace", "stage": "model", "payload": {"sequence": 1}}
            yield {"event": "final", "result": FakeResult()}

    monkeypatch.setattr(agent_routes, "get_agent_app", lambda: FakeAgentApp())

    response = TestClient(app).post(
        "/v1/agent/stream",
        json={"thread_id": "stream-test", "user_input": "hello", "include_trace": True},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    lines = [line for line in response.text.splitlines() if line]
    assert [json.loads(line)["event"] for line in lines] == ["token", "trace", "final"]
