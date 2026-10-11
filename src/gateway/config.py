from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    tailscale_host: str = "127.0.0.1"
    backend_url: str = "http://llm:8080"
    model_name: str = "llm"
    gateway_port: int = 8000
    request_timeout: float = 120.0
    agent_max_iterations: int = 4
    agent_max_context_messages: int = 10
    agent_enable_memory: bool = True
    agent_run_log_path: str = "logs/agent-runs.jsonl"
    langfuse_enabled: bool = False
    langfuse_environment: str = "local"
    langfuse_trace_name: str = "mrp-agent-run"
    langfuse_flush_on_run_end: bool = False
    camera_frame_fixture_path: str | None = None
    iot_provider: str = "stub"
    mqtt_broker_host: str = "mqtt"
    mqtt_broker_port: int = 1883
    mqtt_nodes: str = ""
    mqtt_connect_timeout: float = 10.0
    mqtt_request_timeout: float = 10.0

    @property
    def mqtt_nodes_list(self) -> list[str]:
        """Return configured nodes from MQTT_NODES (comma-separated full paths).

        There is no fallback — nodes are discovered dynamically via ``node/req``.
        """
        if self.mqtt_nodes.strip():
            return [n.strip() for n in self.mqtt_nodes.split(",") if n.strip()]
        return []

    simulated_temperature_c: float = 21.5
    default_temperature: float = 1.0
    default_top_p: float = 0.95
    default_presence_penalty: float = 0.0
    default_top_k: int = 20

    model_config = {"env_file": ".env", "extra": "ignore"}

    @property
    def gateway_base_url(self) -> str:
        return f"http://{self.tailscale_host}:{self.gateway_port}"

    @property
    def llm_backend_v1_url(self) -> str:
        return f"{self.backend_url.rstrip('/')}/v1"


settings = Settings()
