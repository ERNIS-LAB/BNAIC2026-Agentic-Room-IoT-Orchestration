import base64
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Literal

from langchain.tools import tool
from pydantic import BaseModel, Field

from agent.iot.base import CameraFrameProvider, SensorProvider
from agent.iot.sensors import NUMERIC_SENSOR_NAMES, SENSOR_SPECS


class ReadSensorsArgs(BaseModel):
    sensor_names: list[
        Literal["temperature", "temp", "light", "sound", "mic", "camera"]
    ] = Field(
        description=(
            "Sensor names to read. Allowed: temperature, light, sound, camera. "
            "Include 'camera' to get camera frames for occupancy checks. "
            "Example: ['temperature', 'camera']"
        )
    )
    nodes: list[str] | None = Field(
        default=None,
        description=(
            "Specific nodes to query, e.g. ['r_c2001/node_0001']. "
            "Omit to read from all available nodes."
        ),
    )


def _compact_readings(
    batch: dict[str, dict[str, dict[str, Any]]],
    sensor_names: list[str],
    nodes: list[str] | None,
) -> str:
    """Format batch results as a compact JSON string for the LLM."""
    by_node: dict[str, dict[str, Any]] = {}
    for sensor in sensor_names:
        for node, data in (batch.get(sensor) or {}).items():
            if node not in by_node:
                by_node[node] = {}
            # Keep just the value + unit; drop verbose fields
            by_node[node][sensor] = (
                {"error": data["error"]}
                if "error" in data
                else {"value": data.get("value"), "unit": data.get("unit")}
            )
    queried_nodes = nodes or list(by_node.keys())
    return json.dumps(
        {
            "sensors": sensor_names,
            "nodes_queried": queried_nodes,
            "readings": by_node,
        }
    )


def make_read_sensors_tool(
    sensor_provider: SensorProvider,
    camera_provider: CameraFrameProvider | None = None,
):
    @tool(args_schema=ReadSensorsArgs)
    def read_sensors(sensor_names: list[str], nodes: list[str] | None = None):
        """Read one or more sensors from one or more IoT nodes — all in one call.

        Pass ALL sensor names you need in a single call, including 'camera' for
        occupancy checks. Never call this tool more than once per turn.

        Omit *nodes* to query every known node.
        """
        # Split camera out — it's handled separately from numeric sensors.
        wants_camera = "camera" in sensor_names
        numeric_names = [n for n in sensor_names if n != "camera"]

        # Validate numeric sensor names.
        invalid = [n for n in numeric_names if n not in SENSOR_SPECS]
        if invalid:
            return json.dumps(
                {
                    "error": "unknown_sensors",
                    "sensors": invalid,
                    "available": list(NUMERIC_SENSOR_NAMES),
                }
            )

        # Fetch numeric sensors.
        batch: dict[str, dict[str, dict[str, Any]]] = {}
        if numeric_names:
            batch = sensor_provider.fetch_sensors_batch(numeric_names, nodes=nodes)

        result_blocks: list[dict[str, Any]] = []

        # Always include the compact JSON for numeric readings (or an empty
        # summary when only camera was requested).
        json_text = _compact_readings(batch, numeric_names, nodes)
        result_blocks.append({"type": "text", "text": json_text})

        # Fetch camera frames in parallel.
        if wants_camera and camera_provider is not None:
            target_nodes: list[str] = nodes if nodes else sensor_provider.list_nodes()
            if target_nodes:
                camera_futures: dict[Any, str] = {}
                with ThreadPoolExecutor(
                    max_workers=min(len(target_nodes), 4)
                ) as executor:
                    for node in target_nodes:
                        camera_futures[
                            executor.submit(
                                camera_provider.fetch_latest_frame, node=node
                            )
                        ] = node
                    for future in as_completed(camera_futures):
                        node = camera_futures[future]
                        try:
                            frame = future.result()
                            img_b64 = base64.b64encode(frame).decode("utf-8")
                            mime_type = (
                                "image/jpeg"
                                if frame.startswith(b"\xff\xd8")
                                else "image/png"
                            )
                            result_blocks.append(
                                {"type": "text", "text": f"Camera frame from {node}"}
                            )
                            result_blocks.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{img_b64}"
                                    },
                                }
                            )
                        except Exception:
                            result_blocks.append(
                                {
                                    "type": "text",
                                    "text": f"Camera frame from {node}: unavailable",
                                }
                            )

        # If only camera was requested (no numeric sensors), the output is
        # multimodal content blocks.
        if wants_camera:
            return result_blocks

        # Pure numeric query — return the compact JSON string as before.
        return json_text

    return read_sensors
