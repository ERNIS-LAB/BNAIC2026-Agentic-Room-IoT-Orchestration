import argparse
import io
import logging
import re
import sys
import threading
import time

from PIL import Image

from agent.iot import iot_pb2 as pb

logger = logging.getLogger(__name__)

RESPONSE_TOPIC = "+/+/res/#"
NODE_RES_TOPIC = "node/res"


def main() -> None:
    global ready_for_next_publish, stopped
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s - %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Interactively request and decode IoT sensor readings"
    )
    parser.add_argument("--host", default="127.0.0.1", help="MQTT broker host")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    args = parser.parse_args()

    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        sys.exit("paho-mqtt is required: pip install paho-mqtt")

    ready_for_next_publish = True
    stopped = False

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

        if msg.retain:
            return

        ready_for_next_publish = True

        if msg.topic == NODE_RES_TOPIC:
            node_info = pb.Node()
            node_info.ParseFromString(msg.payload)
            node_info = node_info.id
            print(f"Node detected: {node_info}")
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
            image = camera_info.camera
            Image.open(io.BytesIO(image)).show()
            return

        print(f"\nMessage from sensor {sensor}: {sensor_info}")

    def publisher(client: mqtt.Client):
        global ready_for_next_publish, stopped

        while True:
            if stopped:
                break
            if not client.is_connected():
                time.sleep(0.1)
                continue

            if not ready_for_next_publish:
                time.sleep(0.1)
                continue

            if stopped:
                break

            topic = input("Insert topic to publish to: ")
            if topic != "node/req" and not re.compile(
                r"[^/]+/[^/]+/req/(temperature|mic|light|camera)"
            ).match(topic):
                print("Invalid topic")
                continue
            client.publish(topic)
            ready_for_next_publish = False

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, clean_session=True)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.host, args.port)

    t = threading.Thread(target=publisher, args=(client,), daemon=True)
    t.start()

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        logger.info("Stopped")
        stopped = True

    client.loop_stop()
    client.disconnect()


if __name__ == "__main__":
    main()
