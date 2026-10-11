from types import SimpleNamespace

import pytest

from agent.iot.iot_pb2 import Sound, Temperature
from agent.iot.mqtt import MqttConfig, MqttSensorCache


@pytest.mark.parametrize(
    ("sensor", "request_sensor", "response_sensor", "message", "value"),
    [
        (
            "temperature",
            "temperature",
            "temperature",
            Temperature(temperature=23.5),
            23.5,
        ),
        ("temp", "temperature", "temperature", Temperature(temperature=23.5), 23.5),
        ("sound", "mic", "mic", Sound(sound=-42), -42),
        ("mic", "mic", "mic", Sound(sound=-42), -42),
    ],
)
def test_requests_match_firmware_topics_and_decode_responses(
    sensor, request_sensor, response_sensor, message, value
):
    cache = MqttSensorCache(MqttConfig(host="localhost", nodes=("room/node",)))

    class Client:
        def publish(self, topic, payload):
            assert topic == f"room/node/req/{request_sensor}"
            assert payload == b""
            cache._on_message(
                self,
                None,
                SimpleNamespace(
                    topic=f"room/node/res/{response_sensor}",
                    payload=message.SerializeToString(),
                ),
            )

    cache._client = Client()
    assert cache.request_sensor(sensor)["value"] == value
    assert cache._pending == {}


def test_request_timeout_clears_pending_state():
    cache = MqttSensorCache(MqttConfig(host="localhost", nodes=("room/node",)))
    cache._client = SimpleNamespace(publish=lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="Timed out"):
        cache.request_sensor("temperature", timeout=0)
    assert cache._pending == {}


def test_start_subscribes_only_after_connection(monkeypatch):
    import paho.mqtt.client as mqtt

    class Client:
        connected = False

        def connect(self, *args, **kwargs):
            pass

        def loop_start(self):
            self.connected = True
            self.on_connect(self, None, None, 0)

        def subscribe(self, topics):
            assert self.connected
            assert topics == [("+/+/res/#", 1), ("node/res", 1)]
            self.on_subscribe(self, None, 1, [1, 1])

    monkeypatch.setattr(mqtt, "Client", lambda *args, **kwargs: Client())
    cache = MqttSensorCache(MqttConfig(host="localhost"))
    cache.start()
    assert cache._client.connected
