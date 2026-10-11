"""MQTT-backed IoT providers."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Event, Lock
from typing import Any

from .base import CameraFrameProvider, SensorProvider
from .sensors import decode_sensor_message, node_from_topic

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MqttConfig:
    host: str
    port: int = 1883
    nodes: tuple[str, ...] = ()
    connect_timeout: float = 10.0
    request_timeout: float = 5.0

    # Subscribe to all rooms and nodes (room-agnostic).
    subscribe_topic: str = "+/+/res/#"

    def request_topic(self, sensor: str, node: str | None = None) -> str:
        """Topic the agent publishes to in order to trigger a sensor reading.

        *node* must be a full path like ``r2001/n1`` (as returned by
        discovery).
        """
        target = node or (self.nodes[0] if self.nodes else "")
        if not target:
            raise RuntimeError(
                "No node specified and no nodes configured. "
                "Run discover_iot_nodes first to find available nodes."
            )
        return f"{target}/req/{sensor}"


class MqttSensorCache:
    """Requests sensor readings on demand and caches the response for agent tools.

    Cache is keyed by (node, sensor) to support multiple IoT nodes per room.
    """

    def __init__(self, config: MqttConfig) -> None:
        self._config = config
        self._lock = Lock()
        self._publish_lock = Lock()  # paho-mqtt Client.publish() is not thread-safe
        self._latest: dict[str, dict[str, Any]] = {}
        self._pending: dict[str, Event] = {}
        self._client = None
        self._discovered_nodes: list[str] = []
        self._persistent_nodes: set[str] = set()  # union of all discoveries

    def start(self) -> None:
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:
            raise RuntimeError(
                "MQTT IoT provider requires the 'paho-mqtt' package. "
                "Keep IOT_PROVIDER=stub or install the dependency before enabling MQTT."
            ) from exc

        connected = Event()
        subscribed = Event()

        def _on_start_connect(client, userdata, flags, reason_code, properties=None):
            if reason_code == 0:
                self._on_connect(client, userdata, flags, reason_code, properties)
                connected.set()

        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.on_connect = _on_start_connect
        client.on_message = self._on_message
        # Subscribe after CONNACK; pre-connection subscribe calls are rejected.
        client.on_subscribe = (
            lambda c, u, mid, granted, p=None: subscribed.set()  # noqa: E731
        )
        client.connect(
            self._config.host,
            self._config.port,
            keepalive=int(self._config.connect_timeout),
        )
        client.loop_start()

        if not connected.wait(timeout=5.0):
            raise RuntimeError("MQTT connection timed out")
        if not subscribed.wait(timeout=2.0):
            logger.warning(
                "MQTT initial subscription not confirmed — continuing anyway"
            )
        # Restore on_connect for automatic reconnects.
        client.on_connect = self._on_connect
        self._client = client
        logger.info(
            "MQTT sensor cache connected to %s:%s, subscribed to %s",
            self._config.host,
            self._config.port,
            self._config.subscribe_topic,
        )

    def stop(self) -> None:
        if self._client is None:
            return
        self._client.loop_stop()
        self._client.disconnect()
        self._client = None

    # Camera JPEG frames are large and take ~30 s to transfer over MQTT.
    _CAMERA_TIMEOUT = 45.0

    def latest_camera_frame(self, node: str | None = None) -> bytes:
        reading = self.request_sensor("camera", node=node, timeout=self._CAMERA_TIMEOUT)
        image = reading.get("image") if isinstance(reading.get("image"), dict) else {}
        data = image.get("data")
        if not isinstance(data, str):
            raise RuntimeError(
                "Latest MQTT camera reading does not contain image data."
            )

        import base64

        return base64.b64decode(data)

    def request_sensor(
        self,
        sensor_name: str,
        node: str | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Publish a request for *sensor_name* and block until the response arrives."""
        if self._client is None:
            raise RuntimeError("MQTT client is not started.")

        normalized = sensor_name.strip().lower()
        normalized = {"temp": "temperature", "mic": "sound"}.get(normalized, normalized)
        target_node = node or (self._config.nodes[0] if self._config.nodes else None)
        if not target_node:
            raise RuntimeError(
                "No node specified and no nodes configured. "
                "Run discover_iot_nodes first to find available nodes."
            )

        cache_key = f"{target_node}/{normalized}"
        event = Event()
        with self._lock:
            self._latest.pop(cache_key, None)
            self._pending[cache_key] = event

        # Match the request topics implemented by the firmware.
        _REQ_ALIASES = {"sound": "mic"}
        req_sensor = _REQ_ALIASES.get(normalized, normalized)
        topic = self._config.request_topic(req_sensor, node=target_node)
        with self._publish_lock:
            self._client.publish(topic, payload=b"")
        logger.debug("Published sensor request to %s", topic)

        effective_timeout = (
            timeout if timeout is not None else self._config.request_timeout
        )
        if not event.wait(timeout=effective_timeout):
            with self._lock:
                self._pending.pop(cache_key, None)
            raise RuntimeError(
                f"Timed out waiting for MQTT response for sensor {normalized!r} "
                f"on node {target_node!r} (topic: {self._config.subscribe_topic})"
            )

        with self._lock:
            self._pending.pop(cache_key, None)
            return dict(self._latest[cache_key])

    # ------------------------------------------------------------------
    # Batch sensor reads (parallelised with ThreadPoolExecutor)
    # ------------------------------------------------------------------

    def _ensure_connected(self) -> None:
        """Verify the MQTT client is healthy; reconnect only if necessary.

        Full-disconnect-and-reconnect was too disruptive — it destroys the
        broker session, loses in-flight subscriptions, and causes a window
        where firmware responses are silently dropped.  Instead, only
        reconnect when ``is_connected()`` reports the client as dead.
        """
        if self._client is None:
            raise RuntimeError("MQTT client is not started.")

        # Fast path: client is still connected according to paho.
        if self._client.is_connected():
            return

        # The client died — full reconnect.  Serialise with publish_lock to
        # prevent concurrent request_sensor() calls from publishing to a
        # half-replaced client.
        with self._publish_lock:
            # Double-check after acquiring the lock.
            if self._client.is_connected():
                return

            logger.info("MQTT client disconnected; reconnecting…")
            try:
                self._client.disconnect()
            except Exception:
                pass
            try:
                self._client.loop_stop()
            except Exception:
                pass

            import paho.mqtt.client as mqtt

            connected = Event()
            subscribed = Event()

            def _on_first_connect(
                client, userdata, flags, reason_code, properties=None
            ):
                if reason_code == 0:
                    self._on_connect(client, userdata, flags, reason_code, properties)
                    connected.set()

            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            client.on_connect = _on_first_connect
            client.on_message = self._on_message
            client.on_subscribe = (
                lambda c, u, mid, granted, p=None: subscribed.set()  # noqa: E731
            )
            client.connect(
                self._config.host,
                self._config.port,
                keepalive=int(self._config.connect_timeout),
            )
            client.loop_start()

            if not connected.wait(timeout=5.0):
                logger.warning("MQTT reconnect timed out — continuing anyway")
            if not subscribed.wait(timeout=2.0):
                logger.warning("MQTT subscription not confirmed — continuing anyway")
            client.on_connect = self._on_connect
            self._client = client

        # If we have no nodes, run a fresh discovery.
        if (
            not self._discovered_nodes
            and not self._persistent_nodes
            and not self._config.nodes
        ):
            try:
                discovered = self.discover_nodes(timeout=5.0)
                logger.info("Post-reconnect discovery: %d node(s)", len(discovered))
            except Exception:
                logger.warning("Post-reconnect discovery failed", exc_info=True)

    def _resolve_nodes(self, nodes: list[str] | None) -> list[str]:
        """Return the list of nodes to query, resolving *None* to all known.

        Prefers nodes found by the most recent discovery.  Falls back to
        previously-seen nodes only when discovery found nothing at all
        (e.g. all boards are temporarily unreachable).
        """
        if nodes:
            return list(nodes)

        current = set(self._discovered_nodes)
        remembered = self._persistent_nodes - current
        configured = set(self._config.nodes) - current - remembered

        # If discovery found nodes this session, trust it — don't
        # query nodes that didn't respond to discovery.
        if current:
            return sorted(current)

        # Discovery found nothing.  Try once more in case nodes came
        # online since startup.
        if not current and not remembered and not configured:
            logger.info("No known nodes; running fresh discovery...")
            try:
                discovered = self.discover_nodes(timeout=5.0)
                if discovered:
                    return list(discovered)
            except Exception:
                logger.warning("Fresh discovery failed", exc_info=True)

        # Last resort: nodes we've seen before + configured fallback.
        fallback = sorted(remembered | configured)
        if fallback:
            logger.info(
                "Discovery found nothing; falling back to %d remembered node(s): %s",
                len(fallback),
                fallback,
            )
            return fallback

        raise RuntimeError(
            "No nodes available. The IoT nodes may be offline or not "
            "connected to the MQTT broker. Check that the devices are "
            "powered on and on the same network."
        )

    def request_sensors_batch(
        self,
        sensor_names: list[str],
        nodes: list[str] | None = None,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Read *sensor_names* from *nodes* in parallel.

        If *nodes* is ``None``, all known nodes are queried.

        Returns a dict keyed by sensor_name, then by node name::

            {"temperature": {"r_c2001/node_0001": {...}, ...}, "light": {...}}
        """
        self._ensure_connected()
        normalized_sensors = [s.strip().lower() for s in sensor_names]
        target_nodes = self._resolve_nodes(nodes)

        results: dict[str, dict[str, dict[str, Any]]] = {
            sensor: {} for sensor in normalized_sensors
        }
        total_tasks = len(normalized_sensors) * len(target_nodes)
        if total_tasks == 0:
            return results

        with ThreadPoolExecutor(max_workers=min(total_tasks, 2)) as executor:
            futures: dict[Any, tuple[str, str]] = {}
            for sensor in normalized_sensors:
                for node in target_nodes:
                    future = executor.submit(self.request_sensor, sensor, node=node)
                    futures[future] = (sensor, node)

            for future in as_completed(futures):
                sensor, node = futures[future]
                try:
                    results[sensor][node] = future.result()
                except RuntimeError:
                    logger.warning(
                        "Node %r did not respond to %r request", node, sensor
                    )
                    results[sensor][node] = {
                        "error": "timeout",
                        "sensor": sensor,
                        "node": node,
                    }
        return results

    # ------------------------------------------------------------------
    # Node helpers
    # ------------------------------------------------------------------

    def list_known_nodes(self) -> list[str]:
        """Return all nodes ever seen (discovered + configured + persistent)."""
        all_nodes = set(self._discovered_nodes)
        all_nodes.update(self._persistent_nodes)
        all_nodes.update(self._config.nodes)
        return sorted(all_nodes)

    # ------------------------------------------------------------------
    # Node discovery
    # ------------------------------------------------------------------

    def discover_nodes(self, timeout: float = 3.0) -> list[str]:
        """Publish a discovery request and collect node IDs from responses.

        Publishes to ``node/req`` (global topic the firmware subscribes to).
        Firmware responds on ``node/res`` with a protobuf-encoded
        ``iot_Node`` message whose ``id`` field is the full node path
        (e.g. ``r2001/n1``).
        """
        if self._client is None:
            raise RuntimeError("MQTT client is not started.")

        discovered: list[str] = []
        discovery_lock = Lock()

        def _on_discovery(_client, _userdata, msg) -> None:
            logger.debug(
                "Discovery response on topic %s: %d bytes", msg.topic, len(msg.payload)
            )
            node_name = _decode_iot_node(msg.payload)
            if node_name:
                logger.info("Discovered node: %s", node_name)
                with discovery_lock:
                    if node_name not in discovered:
                        discovered.append(node_name)
            else:
                logger.warning(
                    "Failed to decode discovery response on %s (payload: %r)",
                    msg.topic,
                    msg.payload[:80],
                )

        self._client.message_callback_add("node/res", _on_discovery)
        try:
            time.sleep(0.15)  # let the subscription round-trip before publishing
            with self._publish_lock:
                self._client.publish("node/req", payload=b"")
            logger.info("Published node discovery request to node/req")

            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                time.sleep(0.1)
            time.sleep(0.3)  # grace period for late responses
        finally:
            self._client.message_callback_remove("node/res")

        # Cache for the current session and accumulate into persistent set
        if discovered:
            self._discovered_nodes = list(discovered)
            self._persistent_nodes.update(discovered)

        logger.info("Discovered %d nodes: %s", len(discovered), discovered)
        return discovered

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_connect(
        self, client, userdata, flags, reason_code, properties=None
    ) -> None:
        if reason_code != 0:
            logger.warning("MQTT broker connection returned code %s", reason_code)
            return
        client.subscribe([(self._config.subscribe_topic, 1), ("node/res", 1)])

    def _on_message(self, client, userdata, msg) -> None:
        # Only process response topics: room/node/res/sensor  or  node/res
        if "/res/" not in msg.topic and msg.topic != "node/res":
            return

        try:
            reading = decode_sensor_message(msg.topic, msg.payload)
        except Exception:
            logger.exception(
                "Failed to decode MQTT sensor message from topic %s", msg.topic
            )
            return

        sensor = str(reading.get("sensor") or "")
        if not sensor or reading.get("known") is False:
            return

        # Use full room/node path for cache key (consistent with discovery)
        node = str(reading.get("node") or "")
        if not node:
            node = node_from_topic(msg.topic)
        if not node:
            return

        cache_key = f"{node}/{sensor}"

        with self._lock:
            self._latest[cache_key] = dict(reading)
            if cache_key in self._pending:
                self._pending[cache_key].set()


# ------------------------------------------------------------------
# Provider wrappers
# ------------------------------------------------------------------


class MqttCameraFrameProvider(CameraFrameProvider):
    def __init__(self, cache: MqttSensorCache) -> None:
        self._cache = cache

    def fetch_latest_frame(self, node: str | None = None) -> bytes:
        return self._cache.latest_camera_frame(node=node)


class MqttSensorProvider(SensorProvider):
    def __init__(self, cache: MqttSensorCache) -> None:
        self._cache = cache

    def fetch_latest_sensor(
        self, sensor_name: str, node: str | None = None
    ) -> dict[str, Any]:
        return self._cache.request_sensor(sensor_name, node=node)

    def fetch_sensors_batch(
        self, sensor_names: list[str], nodes: list[str] | None = None
    ) -> dict[str, dict[str, dict[str, Any]]]:
        return self._cache.request_sensors_batch(sensor_names, nodes=nodes)

    def list_nodes(self) -> list[str]:
        return self._cache.list_known_nodes()

    def discover_nodes(self, timeout: float = 3.0) -> list[str]:
        return self._cache.discover_nodes(timeout=timeout)


# ------------------------------------------------------------------
# Protobuf helpers
# ------------------------------------------------------------------


def _decode_iot_node(payload: bytes) -> str | None:
    """Decode a protobuf-encoded ``iot_Node`` message.

    Wire format for ``required string id = 1``:

        tag 0x0A | varint-length | UTF-8 bytes
    """
    if not payload or len(payload) < 2 or payload[0] != 0x0A:
        return None

    try:
        remaining = payload[1:]
        length = 0
        shift = 0
        for i, byte in enumerate(remaining):
            length |= (byte & 0x7F) << shift
            shift += 7
            if not (byte & 0x80):
                remaining = remaining[i + 1 :]
                break
            if shift > 35:
                return None

        return remaining[:length].decode("utf-8", errors="replace")
    except (IndexError, ValueError):
        return None
