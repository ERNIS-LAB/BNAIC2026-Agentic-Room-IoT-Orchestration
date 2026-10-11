"""LangChain v1 agent graph construction."""

from typing import Any

from langchain.agents import create_agent
from langchain.agents.structured_output import (
    AutoStrategy,
    ProviderStrategy,
    ToolStrategy,
)
from langchain_core.language_models.chat_models import BaseChatModel

from agent.graph.middleware import TrimMessagesMiddleware
from agent.graph.prompts import SYSTEM_PROMPT

# Union type matching create_agent's response_format parameter.
ResponseFormatArg = (
    ToolStrategy[Any]
    | ProviderStrategy[Any]
    | AutoStrategy[Any]
    | type[Any]
    | dict[str, Any]
    | None
)


def build_agent_graph(
    model: BaseChatModel,
    tools: list,
    *,
    checkpointer=None,
    context_schema=None,
    max_context_messages: int = 10,
    response_format: ResponseFormatArg = None,
):
    """Build a compiled LangGraph agent.

    *response_format* enables structured output.  Pass a
    :class:`ToolStrategy`, :class:`ProviderStrategy`, or a raw Pydantic
    model / JSON schema dict — LangChain auto-detects the best strategy.
    """
    return create_agent(
        model,
        tools=tools,
        checkpointer=checkpointer,
        middleware=[
            TrimMessagesMiddleware(max_tokens=max_context_messages * 200),
        ],
        system_prompt=SYSTEM_PROMPT,
        context_schema=context_schema,
        response_format=response_format,
    )
