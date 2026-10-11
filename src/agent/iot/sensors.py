"""Central sensor registry and payload decoding."""

import base64
import io
from dataclasses import dataclass
from typing import Any, Literal

from .iot_pb2 import (  # type: ignore[attr-defined]  # protobuf dynamic attrs
    Camera,
    Light,
    Sound,
    Temperature,
)

SensorKind = Literal["numeric", "image"]


@dataclass(frozen=True)
class SensorSpec:
    name: str
    msg_class: type
    field_name: str
    unit: str | None
    kind: SensorKind = "numeric"


SENSOR_SPECS: dict[str, SensorSpec] = {
    "temperature": SensorSpec("temperature", Temperature, "temperature", "C"),
    "temp": SensorSpec("temperature", Temperature, "temperature", "C"),
    "light": SensorSpec("light", Light, "light", "lux"),
    "sound": SensorSpec("sound", Sound, "sound", "dBFS"),
    "mic": SensorSpec("sound", Sound, "sound", "dBFS"),
    "camera": SensorSpec("camera", Camera, "camera", None, kind="image"),
}

NUMERIC_SENSOR_NAMES = tuple(
    name for name, spec in SENSOR_SPECS.items() if spec.kind == "numeric"
)


def sensor_name_from_topic(topic: str) -> str:
    return topic.rstrip("/").split("/")[-1]


def node_from_topic(topic: str) -> str:
    """Extract the full node path from an MQTT topic.

    Returns ``room/node`` (e.g. ``r2001/n1``) for topics like
    ``r2001/n1/res/camera``, matching the format used by node
    discovery.
    """
    parts = topic.rstrip("/").split("/")
    # Topic format: room/node/res/sensor
    if len(parts) >= 4 and parts[-2] == "res":
        return f"{parts[-4]}/{parts[-3]}"
    return ""


def _extract_jpeg(payload: bytes) -> bytes | None:
    """Extract JPEG data from a raw payload that may contain leading/trailing garbage."""
    idx = payload.find(b"\xff\xd8")
    if idx < 0:
        return None
    data = payload[idx:]
    eoi = data.rfind(b"\xff\xd9")
    if eoi >= 0:
        data = data[: eoi + 2]
    return data


def decode_sensor_message(topic: str, payload: bytes) -> dict[str, Any]:
    sensor_name = sensor_name_from_topic(topic)
    node = node_from_topic(topic)
    spec = SENSOR_SPECS.get(sensor_name)
    if spec is None:
        return {
            "sensor": sensor_name or "unknown",
            "node": node,
            "source": topic,
            "known": False,
        }

    if spec.kind == "image":
        # Firmware sends raw JPEG; detect it even when protobuf parse succeeds
        msg = spec.msg_class()
        msg.ParseFromString(payload)
        value = getattr(msg, spec.field_name)

        # If the parsed bytes are JPEG, handle them directly
        if isinstance(value, bytes) and value[:2] == b"\xff\xd8":
            return {
                "sensor": spec.name,
                "node": node,
                "source": topic,
                "image": jpeg_to_b64(value),
            }

        # Otherwise try to convert as raw pixel data
        try:
            return {
                "sensor": spec.name,
                "node": node,
                "source": topic,
                "image": raw_to_png(value),
            }
        except Exception:
            jpeg = _extract_jpeg(payload)
            if jpeg is None:
                raise
            return {
                "sensor": spec.name,
                "node": node,
                "source": topic,
                "image": jpeg_to_b64(jpeg),
            }

    msg = spec.msg_class()
    msg.ParseFromString(payload)
    value = getattr(msg, spec.field_name)

    return {
        "sensor": spec.name,
        "node": node,
        "source": topic,
        "value": value,
        "unit": spec.unit,
    }


def jpeg_to_b64(jpeg_bytes: bytes) -> dict[str, Any]:
    """Convert JPEG bytes to a base64-encoded data URI dict for VLM input."""
    return {
        "mime_type": "image/jpeg",
        "data": base64.b64encode(jpeg_bytes).decode("utf-8"),
    }


def raw_to_png(
    raw_data: bytes | list[int], width: int = 640, height: int = 480
) -> dict[str, Any]:
    """Convert raw camera data to greyscale PNG bytes for VLM input.

    Handles both legacy UYVY422 bytes and nanopb firmware RGB565 uint32 lists.
    """
    import numpy as np
    from PIL import Image

    if isinstance(raw_data, list):
        # RGB565 uint32 list from nanopb firmware (repeated uint32)
        pixels = np.array(raw_data, dtype=np.uint32)
        r = ((pixels >> 11) & 0x1F).astype(np.uint8) * 255 // 31
        g = ((pixels >> 5) & 0x3F).astype(np.uint8) * 255 // 63
        b = (pixels & 0x1F).astype(np.uint8) * 255 // 31
        grey = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)

        if len(grey) >= width * height:
            grey = grey[: width * height]
        grey = grey.reshape((height, width))
        img = Image.fromarray(grey, "L")
    else:
        # Legacy UYVY422 raw bytes
        data = np.frombuffer(raw_data[: width * height * 2], dtype=np.uint8).reshape(
            (height, width // 2, 4)
        )
        y0 = data[:, :, 1].astype(np.float32)
        y1 = data[:, :, 3].astype(np.float32)
        y_full = np.empty((height, width), dtype=np.float32)
        y_full[:, 0::2] = y0
        y_full[:, 1::2] = y1
        img = Image.fromarray(np.clip(y_full, 0, 255).astype(np.uint8), "L")

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    png_bytes = buffer.getvalue()

    return {
        "mime_type": "image/png",
        "data": base64.b64encode(png_bytes).decode("utf-8"),
    }
