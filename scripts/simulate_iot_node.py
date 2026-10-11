"""Simulate an IoT node by responding to MQTT sensor requests.

Listens on room_ls/+/req/# and responds with protobuf-encoded sensor readings.
Run this to test the agent without physical STM32 hardware.
"""

import argparse
import logging

from agent.iot.iot_pb2 import Node, Sound, Temperature
from agent.iot.sensors import sensor_name_from_topic

logger = logging.getLogger(__name__)

RESPONSE_TOPIC = "room_ls/node_0/res/{sensor}"
SUBSCRIBE_TOPIC = "room_ls/+/req/#"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="Simulate IoT sensor node over MQTT")
    parser.add_argument("--host", default="127.0.0.1", help="MQTT broker host")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument(
        "--temperature",
        type=float,
        default=23.5,
        help="Simulated temperature in Celsius",
    )
    parser.add_argument(
        "--sound", type=int, default=-45, help="Simulated sound level in dBFS"
    )
    args = parser.parse_args()

    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        import sys

        sys.exit("paho-mqtt is required: pip install paho-mqtt")

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc != 0:
            logger.error("Broker connection failed (rc=%s)", rc)
            return
        client.subscribe(SUBSCRIBE_TOPIC)
        client.subscribe("node/req")
        logger.info(
            "Connected to %s:%s, subscribed to %s",
            args.host,
            args.port,
            SUBSCRIBE_TOPIC,
        )

    def on_message(client, userdata, msg):
        if msg.topic == "node/req":
            client.publish("node/res", Node(id="room_ls/node_0").SerializeToString())
            return
        sensor = sensor_name_from_topic(msg.topic)
        logger.info("Request: %s → sensor=%s", msg.topic, sensor)

        if sensor == "temperature":
            message = Temperature()
            message.temperature = args.temperature
        elif sensor == "sound" or sensor == "mic":
            message = Sound()
            message.sound = args.sound
        else:
            logger.info("Unknown sensor %r — ignoring", sensor)
            return

        payload = message.SerializeToString()
        topic = RESPONSE_TOPIC.format(sensor=sensor)
        client.publish(topic, payload=payload)
        logger.info(
            "Response: %s (%d bytes, value=%s)",
            topic,
            len(payload),
            message.temperature if sensor == "temperature" else message.sound,
        )

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
