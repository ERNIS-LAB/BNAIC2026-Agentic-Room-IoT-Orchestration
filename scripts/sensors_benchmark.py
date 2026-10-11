import argparse
import logging
import re
import sys
import threading
import time

from agent.iot import iot_pb2 as pb

logger = logging.getLogger(__name__)

RESPONSE_TOPIC = "rls/+/res/#"
NODE_RES_TOPIC = "node/res"


def main() -> None:
    global ready_for_next_publish
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s - %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Collect MQTT sensor latency measurements"
    )
    parser.add_argument("--host", default="127.0.0.1", help="MQTT broker host")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    args = parser.parse_args()

    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        sys.exit("paho-mqtt is required: pip install paho-mqtt")

    ready_for_next_publish = True

    def on_connect(client: mqtt.Client, userdata, flags, rc, properties):
        if rc != 0:
            logger.error("Broker connection failed (rc=%s)", rc)
            return
        client.subscribe(RESPONSE_TOPIC, qos=2)
        client.subscribe(NODE_RES_TOPIC)
        logger.info(
            "Connected to %s:%s, subscribed to %s", args.host, args.port, RESPONSE_TOPIC
        )

    def on_message(client, userdata, msg: mqtt.MQTTMessage):
        global ready_for_next_publish

        if msg.topic == NODE_RES_TOPIC:
            node_info = pb.Node()
            node_info.ParseFromString(msg.payload)
            node_info = node_info.id
            print(f"Node detected: {node_info}")
            ready_for_next_publish = True
            return

        sensor = re.compile(r".+/.+/.+/(.+)").findall(msg.topic)[0]
        sensor_info = None
        if sensor == "temperature":
            sensor_info = pb.Temperature()
            sensor_info.ParseFromString(msg.payload)
            sensor_info = sensor_info.temperature
        elif sensor == "mic":
            sensor_info = pb.Sound()
            sensor_info.ParseFromString(msg.payload)
            sensor_info = sensor_info.sound
        elif sensor == "light":
            sensor_info = pb.Light()
            sensor_info.ParseFromString(msg.payload)
            sensor_info = sensor_info.light
        elif sensor == "camera":
            camera_info = pb.Camera()
            camera_info.ParseFromString(msg.payload)
            ready_for_next_publish = True
            return

        print(f"\nMessage from sensor {sensor}: {sensor_info}")
        ready_for_next_publish = True

    def publisher(client: mqtt.Client):
        global ready_for_next_publish

        counter = 0
        while True:
            if not client.is_connected():
                time.sleep(0.01)
                continue

            if not ready_for_next_publish:
                time.sleep(0.01)
                continue

            print(f"Publishing for iteration {counter}")
            client.publish("rls/n1/req/light")
            ready_for_next_publish = False
            counter += 1

            time.sleep(0.1)

            if counter >= 102:
                break

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, clean_session=True)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.host, args.port)

    t = threading.Thread(target=publisher, args=(client,))
    t.start()

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        logger.info("Stopped")

    t.join()


if __name__ == "__main__":
    main()
