"""Structured run logging for completed agent turns."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from agent.schemas import AgentRunLog

logger = logging.getLogger(__name__)


class AgentRunJsonlLogger:
    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None

    def append(self, run_log: AgentRunLog) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(run_log.to_dict(), ensure_ascii=True) + "\n")
        except OSError:
            logger.warning(
                "Could not write agent run log to %s", self.path, exc_info=True
            )
