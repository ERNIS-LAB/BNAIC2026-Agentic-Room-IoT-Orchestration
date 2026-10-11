"""LangChain middleware for the agent graph."""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
)
from langchain_core.messages import RemoveMessage, trim_messages
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime


class TrimMessagesMiddleware(AgentMiddleware):
    """Keep model calls within a token budget."""

    def __init__(self, max_tokens: int = 4800) -> None:
        super().__init__()
        self.max_tokens = max_tokens

    def before_model(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        messages = state["messages"]
        trimmed = trim_messages(
            messages,
            max_tokens=self.max_tokens,
            token_counter="approximate",
            strategy="last",
            start_on="human",
            allow_partial=False,
        )
        if len(trimmed) == len(messages):
            return None
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *trimmed,
            ]
        }
