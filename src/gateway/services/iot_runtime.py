"""Gateway-owned IoT provider wiring."""

import logging
import time
from functools import lru_cache
from pathlib import Path

from agent.iot import (
    CameraFrameProvider,
    MqttCameraFrameProvider,
    MqttConfig,
    MqttSensorCache,
    MqttSensorProvider,
    SensorProvider,
    StubCameraFrameProvider,
    StubSensorProvider,
)
from gateway.config import settings

logger = logging.getLogger(__name__)


@lru_cache
def get_iot_providers() -> tuple[CameraFrameProvider, SensorProvider]:
    provider = settings.iot_provider.strip().lower()
    if provider == "mqtt":
        nodes = tuple(settings.mqtt_nodes_list)
        cache = MqttSensorCache(
            MqttConfig(
                host=settings.mqtt_broker_host,
                port=settings.mqtt_broker_port,
                nodes=nodes,
                connect_timeout=settings.mqtt_connect_timeout,
                request_timeout=settings.mqtt_request_timeout,
            )
        )
        cache.start()

        # Auto-discover nodes at startup. WiFi-connected boards can be
        # slow to respond, so retry if we only find one node.
        if not nodes:
            for attempt in range(3):
                try:
                    discovered = cache.discover_nodes(timeout=6.0)
                    logger.info(
                        "Startup discovery attempt %d: found %d node(s): %s",
                        attempt + 1,
                        len(discovered),
                        discovered,
                    )
                    if len(discovered) >= 2:
                        break
                except Exception:
                    logger.warning(
                        "Startup discovery attempt %d failed",
                        attempt + 1,
                        exc_info=True,
                    )
                if attempt < 2:
                    time.sleep(2.0)

        return MqttCameraFrameProvider(cache), MqttSensorProvider(cache)

    if provider != "stub":
        logger.warning(
            "Unknown IOT_PROVIDER=%r; falling back to stub providers", provider
        )

    camera = (
        StubCameraFrameProvider(fixture_path=Path(settings.camera_frame_fixture_path))
        if settings.camera_frame_fixture_path
        else StubCameraFrameProvider()
    )
    sensors = StubSensorProvider.with_simulated_temperature(
        value_c=settings.simulated_temperature_c
    )
    return camera, sensors


def close_iot_runtime() -> None:
    if not get_iot_providers.cache_info().currsize:
        return

    providers = get_iot_providers()
    for provider in providers:
        cache = getattr(provider, "_cache", None)
        if isinstance(cache, MqttSensorCache):
            cache.stop()
            break
    get_iot_providers.cache_clear()
