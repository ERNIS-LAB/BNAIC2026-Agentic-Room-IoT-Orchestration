from __future__ import annotations

import argparse
import logging
import sys

from agent.iot.iot_pb2 import Light, Node, Sound, Temperature

logger = logging.getLogger(__name__)

# MQTT topic format follows the real IoT firmware convention:
#   room/node/req/sensor   →  room/node/res/sensor
# Discovery uses the global topic  node/req  →  node/res
RESPONSE_TOPIC = "{room}/{node}/res/{sensor}"

# Ground truth (matches test cases in benchmark_cases.py):
# Room   | temp °C | sound dBFS | light lux | people
# C1.02  |  31.5   |   -42      |   450     |   5
# C1.03  |  23.0   |   -20      |   800     |   0
# C1.04  |  19.5   |   -55      |   200     |   2
NODES_DATA = {
    "C1.02": {
        "temperature": 31.5,
        "sound": -42,
        "light": 450,
    },
    "C1.03": {
        "temperature": 23.0,
        "sound": -20,
        "light": 800,
    },
    "C1.04": {
        "temperature": 19.5,
        "sound": -55,
        "light": 200,
    },
}


def _node_path(room: str) -> str:
    """Return the full node path for a room: room_ls/<room>."""
    return f"room_ls/{room}"


def _encode_node_id(node_id: str) -> bytes:
    """Protobuf-encode a Node(id=node_id) message for discovery responses."""
    msg = Node()
    msg.id = node_id
    return msg.SerializeToString()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="Simulate IoT sensor nodes over MQTT")
    parser.add_argument("--host", default="127.0.0.1", help="MQTT broker host")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    args = parser.parse_args()

    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        sys.exit("paho-mqtt is required: pip install paho-mqtt")

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc != 0:
            logger.error("Broker connection failed (rc=%s)", rc)
            return
        # Subscribe to sensor requests: room/node/req/sensor
        client.subscribe("+/+/req/#")
        # Subscribe to global discovery requests
        client.subscribe("node/req")
        logger.info("Connected to %s:%s", args.host, args.port)

    def on_message(client, userdata, msg):
        # Discovery request
        if msg.topic == "node/req":
            logger.info(
                "Discovery request received, responding with %d nodes", len(NODES_DATA)
            )
            for room in sorted(NODES_DATA):
                payload = _encode_node_id(_node_path(room))
                client.publish("node/res", payload=payload)
                logger.info(
                    "Discovery response: node=%s (%d bytes)",
                    _node_path(room),
                    len(payload),
                )
            return

        # Sensor request: room/node/req/sensor
        parts = msg.topic.split("/")
        if len(parts) < 4 or parts[2] != "req":
            return

        room = parts[1]  # "C1.02"
        sensor_raw = parts[3]  # "temperature" or "temp" (alias)

        # Accept legacy temperature request aliases.
        _SENSOR_ALIASES = {"temp": "temperature"}
        sensor = _SENSOR_ALIASES.get(sensor_raw, sensor_raw)

        logger.info("Request: %s → room=%s, sensor=%s", msg.topic, room, sensor)

        if room not in NODES_DATA:
            logger.warning("Unknown room %r — ignoring", room)
            return

        room_sensors = NODES_DATA[room]
        # Map MQTT request aliases back to sensor names (the agent sends "mic"
        # as a short alias for "sound").
        _SENSOR_ALIASES = {"mic": "sound", "temp": "temperature"}
        resolved = _SENSOR_ALIASES.get(sensor, sensor)
        if resolved not in room_sensors:
            logger.info("Room %s lacks sensor %r — ignoring", room, sensor)
            return

        value = room_sensors[resolved]
        sensor = resolved  # use the canonical name for protobuf encoding

        if sensor == "temperature":
            message = Temperature()
            message.temperature = value
        elif sensor == "sound":
            message = Sound()
            message.sound = int(value)
        elif sensor == "light":
            message = Light()
            message.light = int(value)
        else:
            logger.info("Unknown sensor type %r — add protobuf logic if needed", sensor)
            return

        payload = message.SerializeToString()
        topic = RESPONSE_TOPIC.format(room=parts[0], node=room, sensor=sensor)
        client.publish(topic, payload=payload)
        logger.info("Response: %s (%d bytes, value=%s)", topic, len(payload), value)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.host, args.port, keepalive=60)

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        logger.info("Stopped")


if __name__ == "__main__":
    main()
