"""Optional observability integrations for agent runs."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LangfuseConfig:
    enabled: bool = False
    environment: str = "local"
    trace_name: str = "mrp-agent-run"
    flush_on_run_end: bool = False
    tags: tuple[str, ...] = field(default_factory=tuple)


class LangfuseTracer:
    """Build LangChain callback config for Langfuse when enabled."""

    def __init__(self, config: LangfuseConfig | None = None) -> None:
        self.config = config or LangfuseConfig()
        self._warned_missing_sdk = False

    def langchain_config(self, *, run_id: str, thread_id: str) -> dict[str, Any]:
        if not self.config.enabled:
            return {}

        self._prepare_environment()
        handler = self._callback_handler()
        if handler is None:
            return {}

        tags = self._tags()
        return {
            "callbacks": [handler],
            "run_name": self.config.trace_name,
            "tags": tags,
            "metadata": {
                "agent_run_id": run_id,
                "thread_id": thread_id,
                "langfuse_session_id": thread_id,
                "langfuse_tags": tags,
            },
        }

    def flush(self) -> None:
        if not self.config.enabled or not self.config.flush_on_run_end:
            return

        try:
            from langfuse import get_client
        except ImportError:
            self._warn_missing_sdk()
            return

        try:
            get_client().flush()
        except Exception:
            logger.warning("Could not flush Langfuse events", exc_info=True)

    def _callback_handler(self):
        try:
            from langfuse.langchain import CallbackHandler
        except ImportError:
            self._warn_missing_sdk()
            return None

        try:
            return CallbackHandler()
        except Exception:
            logger.warning("Could not create Langfuse callback handler", exc_info=True)
            return None

    def _prepare_environment(self) -> None:
        # Langfuse SDK v4 uses LANGFUSE_BASE_URL. Some docs and older setups use
        # LANGFUSE_HOST, so accept it as a compatibility alias.
        if not os.environ.get("LANGFUSE_BASE_URL") and os.environ.get("LANGFUSE_HOST"):
            os.environ["LANGFUSE_BASE_URL"] = os.environ["LANGFUSE_HOST"]

    def _tags(self) -> list[str]:
        tags = ("mrp-agent", self.config.environment, *self.config.tags)
        return list(dict.fromkeys(tag for tag in tags if tag))

    def _warn_missing_sdk(self) -> None:
        if self._warned_missing_sdk:
            return
        logger.warning(
            "Langfuse tracing is enabled but the 'langfuse' package is not installed."
        )
        self._warned_missing_sdk = True
