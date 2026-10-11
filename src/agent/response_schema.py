"""Structured response schema for the MRP-2 room-monitoring agent.

The agent uses LangChain's ToolStrategy structured-output mechanism: the
schema is exposed as a synthetic ``AgentResponse`` tool that the model must
call as its final action.  LangChain parses the tool-call arguments into a
validated ``AgentResponse`` instance.
"""

from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    """Final structured answer from the room-monitoring agent."""

    answer: str = Field(
        description=(
            "Direct answer to the user's question. Include exact numeric "
            "values from sensor readings (e.g. '31.5 °C', '-42 dBFS', "
            "'450 lux').  For yes/no questions, use 'YES' or 'NO'.  For "
            "room recommendations, include the room name with reasoning."
        ),
    )
    reasoning: str = Field(
        description=(
            "Step-by-step reasoning: which sensors were consulted, which "
            "nodes were queried, how the values were compared, and why "
            "the conclusion follows.  One or two sentences is sufficient."
        ),
    )
    sensors_used: list[str] = Field(
        default_factory=list,
        description=(
            "Sensor types that were actually consulted to answer the "
            "question: 'temperature', 'sound', 'light', and/or 'camera'."
        ),
    )
    confidence: str = Field(
        default="high",
        description=(
            "Confidence in the answer: 'high' when sensor data directly "
            "supports the conclusion, 'medium' when some inference is "
            "required, 'low' when the answer is uncertain or the agent "
            "must decline."
        ),
    )
