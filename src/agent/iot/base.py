"""IoT boundary interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class CameraFrameProvider(ABC):
    """Contract for retrieving the latest camera frame bytes."""

    @abstractmethod
    def fetch_latest_frame(self, node: str | None = None) -> bytes:
        raise NotImplementedError


class SensorProvider(ABC):
    """Contract for retrieving sensor readings."""

    @abstractmethod
    def fetch_latest_sensor(
        self, sensor_name: str, node: str | None = None
    ) -> dict[str, Any]:
        raise NotImplementedError

    def discover_nodes(self, timeout: float = 3.0) -> list[str]:
        """Discover available IoT nodes. Default: empty list for non-MQTT providers."""
        return []

    def list_nodes(self) -> list[str]:
        """Return the known node list."""
        return []

    def fetch_sensors_batch(
        self, sensor_names: list[str], nodes: list[str] | None = None
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Read *sensor_names* from *nodes*.

        If *nodes* is ``None``, reads from all known nodes.

        Returns a dict keyed by sensor_name, then by node name.
        Default implementation reads each sensor from each node sequentially.
        """
        target_nodes = nodes or self.list_nodes()
        results: dict[str, dict[str, dict[str, Any]]] = {
            sensor: {} for sensor in sensor_names
        }
        for sensor in sensor_names:
            for node in target_nodes:
                try:
                    results[sensor][node] = self.fetch_latest_sensor(sensor, node=node)
                except Exception:
                    results[sensor][node] = {
                        "error": "unavailable",
                        "sensor": sensor,
                        "node": node,
                    }
        return results
