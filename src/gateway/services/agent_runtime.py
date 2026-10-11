from functools import lru_cache

from agent.app import build_agent_app
from gateway.config import settings
from gateway.services.iot_runtime import get_iot_providers


@lru_cache
def get_agent_app():
    camera_provider, sensor_provider = get_iot_providers()
    return build_agent_app(
        settings=settings,
        camera_provider=camera_provider,
        sensor_provider=sensor_provider,
        llm_base_url=settings.llm_backend_v1_url,
    )
