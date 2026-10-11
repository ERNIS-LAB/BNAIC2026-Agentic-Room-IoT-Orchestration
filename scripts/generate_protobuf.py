"""Regenerate shared Python and firmware protobuf bindings."""

import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    proto_dir = os.environ.get("NANOPB_PROTO_DIR")
    if not proto_dir or not (Path(proto_dir) / "nanopb.proto").is_file():
        raise SystemExit(
            "Set NANOPB_PROTO_DIR to the directory containing nanopb.proto."
        )
    protoc = shutil.which("protoc")
    if not protoc:
        raise SystemExit("Install protoc and make it available on PATH.")
    if not shutil.which("protoc-gen-nanopb"):
        raise SystemExit("Install protoc-gen-nanopb and make it available on PATH.")
    python_out = ROOT / "src/agent/iot"
    command = [protoc, f"-I{ROOT}", f"-I{proto_dir}"]
    subprocess.run(
        command + [f"--python_out={python_out}", "iot.proto", "nanopb.proto"],
        cwd=ROOT,
        check=True,
    )
    binding = python_out / "iot_pb2.py"
    binding.write_text(
        binding.read_text()
        .replace(
            "import nanopb_pb2 as nanopb__pb2",
            "from . import nanopb_pb2 as nanopb__pb2",
        )
        .replace(
            "import nanopb_pb2 as nanopb_pb2", "from . import nanopb_pb2 as nanopb_pb2"
        )
    )
    subprocess.run(
        command + [f"--nanopb_out={ROOT / 'MRP-IOT/NetXDuo/App'}", "iot.proto"],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
