"""Stub camera implementation used until the IoT node is available."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import CameraFrameProvider, SensorProvider


@dataclass
class StubCameraFrameProvider(CameraFrameProvider):
    fixture_bytes: bytes | None = None
    fixture_path: Path | None = None

    def fetch_latest_frame(self, node: str | None = None) -> bytes:
        if self.fixture_bytes is not None:
            return self.fixture_bytes
        if self.fixture_path is not None:
            return self.fixture_path.read_bytes()
        raise NotImplementedError(
            "The IoT camera provider is not wired yet. Supply fixture bytes or a fixture "
            "path for tests, or replace the stub with a real provider."
        )


@dataclass
class StubSensorProvider(SensorProvider):
    readings: dict[str, dict[str, Any]]

    @classmethod
    def with_simulated_temperature(cls, value_c: float = 21.5) -> "StubSensorProvider":
        return cls(
            readings={
                "temperature": {
                    "sensor": "temperature",
                    "value": value_c,
                    "unit": "C",
                    "source": "simulated",
                }
            }
        )

    def list_nodes(self) -> list[str]:
        return ["stub/node_0"]

    def fetch_latest_sensor(
        self, sensor_name: str, node: str | None = None
    ) -> dict[str, Any]:
        normalized = sensor_name.strip().lower()
        if normalized not in self.readings:
            available = ", ".join(sorted(self.readings)) or "none"
            raise RuntimeError(
                f"No stub reading configured for {normalized!r}. Available: {available}"
            )
        return dict(self.readings[normalized])
