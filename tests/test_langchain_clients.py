"""Live client compatibility tests for the OpenAI-compatible gateway."""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration

GATEWAY_URL = os.getenv("GATEWAY_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "120"))


def test_langchain_chat_openai_client_handles_context():
    from langchain_openai import ChatOpenAI
    from pydantic import SecretStr

    llm = ChatOpenAI(
        base_url=f"{GATEWAY_URL.rstrip('/')}/v1",
        model="llm",
        timeout=REQUEST_TIMEOUT,
        api_key=SecretStr("not-needed"),
    )

    response = llm.invoke(
        [
            {"role": "system", "content": "Reply with the requested name only."},
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Alice"},
            {"role": "user", "content": "What is my name?"},
        ]
    )

    assert "alice" in str(response.content).lower()
