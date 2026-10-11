from __future__ import annotations

import base64
import os
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGE_PATH = REPO_ROOT / "test_image_room.png"
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "120"))


@pytest.fixture
def gateway_client():
    with httpx.Client(base_url=GATEWAY_URL, timeout=REQUEST_TIMEOUT) as client:
        yield client


def _chat(client: httpx.Client, messages: list) -> dict:
    response = client.post(
        "/v1/chat/completions",
        json={"model": "llm", "messages": messages},
    )
    response.raise_for_status()
    return response.json()


def _assert_chat_completion(body: dict) -> str:
    assert body["object"] == "chat.completion"
    assert body["model"]
    assert body["choices"][0]["message"]["role"] == "assistant"
    content = body["choices"][0]["message"]["content"]
    assert isinstance(content, str)
    assert content.strip()
    return content


def test_health_and_model_catalog(gateway_client):
    health = gateway_client.get("/health")
    health.raise_for_status()
    health_body = health.json()
    assert health_body["status"] in {"healthy", "degraded"}
    assert len(health_body["backends"]) == 1
    assert health_body["backends"][0]["name"] == "llm"
    assert {"url", "healthy"} <= set(health_body["backends"][0])

    models = gateway_client.get("/v1/models")
    models.raise_for_status()
    assert [model["id"] for model in models.json()["data"]] == ["llm"]


def test_text_chat_completion_contract(gateway_client):
    body = _chat(
        gateway_client,
        [{"role": "user", "content": "What is 2 + 2? Reply with just the number."}],
    )

    _assert_chat_completion(body)
    if "usage" in body:
        assert body["usage"]["total_tokens"] >= body["usage"]["completion_tokens"]


def test_multi_turn_context(gateway_client):
    body = _chat(
        gateway_client,
        [
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Hello Alice!"},
            {"role": "user", "content": "What is my name? Reply with just the name."},
        ],
    )

    assert "alice" in _assert_chat_completion(body).lower()


@pytest.mark.skipif(
    not TEST_IMAGE_PATH.is_file(), reason="test_image_room.png not found"
)
def test_image_chat_completion_contract(gateway_client):
    image = base64.b64encode(TEST_IMAGE_PATH.read_bytes()).decode("utf-8")
    body = _chat(
        gateway_client,
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in one sentence."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image}"},
                    },
                ],
            }
        ],
    )

    assert len(_assert_chat_completion(body)) >= 5


@pytest.mark.parametrize(
    ("payload", "expected_status"),
    [
        (
            {
                "model": "nonexistent-model",
                "messages": [{"role": "user", "content": "hello"}],
            },
            404,
        ),
        ({"messages": [{"role": "user", "content": "hello"}]}, 422),
        ({"model": "llm"}, 422),
    ],
)
def test_chat_request_errors(gateway_client, payload, expected_status):
    response = gateway_client.post("/v1/chat/completions", json=payload)

    assert response.status_code == expected_status
    assert response.json()


def test_agent_query_contract(gateway_client):
    response = gateway_client.post(
        "/v1/agent/query",
        json={
            "thread_id": "gateway-agent-smoke",
            "user_input": "What is 2 + 2? Answer in one short sentence.",
        },
    )

    response.raise_for_status()
    body = response.json()
    assert body["thread_id"] == "gateway-agent-smoke"
    assert body["final_answer"].strip()
    assert isinstance(body["tool_observations"], list)
    assert body["run_log"]["thread_id"] == "gateway-agent-smoke"
