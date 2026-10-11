#!/usr/bin/env python3
"""Build the STM32U585 IoT firmware from the command line.

Uses the ARM GCC toolchain bundled with STM32CubeProgrammer.
Set NODE_NAME via the environment variable, e.g.:

    NODE_NAME=r2001/n1 python3 build_firmware.py
    NODE_NAME=r1001/n1 python3 build_firmware.py
"""

import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
TOOLCHAIN = (
    Path.home() / ".local/share/stm32cube/bundles/gnu-tools-for-stm32/14.3.1+st.2/bin"
)
BUILD_DIR = PROJECT / "build"

TOOLCHAIN = Path(os.environ.get("ARM_TOOLCHAIN_DIR", str(TOOLCHAIN)))
TOOL_SUFFIX = ".exe" if os.name == "nt" else ""
CC = TOOLCHAIN / f"arm-none-eabi-gcc{TOOL_SUFFIX}"
OBJCOPY = TOOLCHAIN / f"arm-none-eabi-objcopy{TOOL_SUFFIX}"
SIZE = TOOLCHAIN / f"arm-none-eabi-size{TOOL_SUFFIX}"

# ── target ──────────────────────────────────────────────────────────
MCU_FLAGS = ["-mcpu=cortex-m33", "-mthumb", "-mfpu=fpv5-sp-d16", "-mfloat-abi=hard"]
COMMON_FLAGS = MCU_FLAGS + ["-ffunction-sections", "-fdata-sections", "-Wall", "-g3"]

DEFINES = [
    "DEBUG",
    "PB_FIELD_32BIT",
    "NX_INCLUDE_USER_DEFINE_FILE",
    "TX_INCLUDE_USER_DEFINE_FILE",
    "TX_SINGLE_MODE_NON_SECURE=1",
    "USE_HAL_DRIVER",
    "STM32U585xx",
]

INCLUDES = [
    "Core/Inc",
    "NetXDuo/App",
    "AZURE_RTOS/App",
    "Drivers/STM32U5xx_HAL_Driver/Inc",
    "Drivers/STM32U5xx_HAL_Driver/Inc/Legacy",
    "Drivers/CMSIS/Device/ST/STM32U5xx/Include",
    "Drivers/CMSIS/Include",
    "Middlewares/ST/netxduo/common/inc",
    "Middlewares/ST/netxduo/ports/cortex_m33/gnu/inc",
    "Middlewares/ST/netxduo/addons/mqtt",
    "Middlewares/ST/netxduo/addons/dhcp",
    "Middlewares/ST/netxduo/addons/dns",
    "Middlewares/ST/netxduo/addons/sntp",
    "Middlewares/ST/threadx/common/inc",
    "Middlewares/ST/threadx/ports/cortex_m33/gnu/inc",
    "Drivers/BSP/Components/Common",
    "Drivers/BSP/Components/ov5640",
    "Drivers/BSP/Components/mx_wifi",
    "Drivers/BSP/Components/mx_wifi/core",
    "Drivers/BSP/B-U585I-IOT02A",
    "NetXDuo/Target",
    "Middlewares/ST/netxduo/common/drivers/wifi/mxchip",
]

LINKER_SCRIPT = PROJECT / "STM32U585AIIXQ_FLASH.ld"
LINKER_FLAGS = MCU_FLAGS + [
    "-T",
    str(LINKER_SCRIPT),
    "-Wl,--gc-sections",
    "--specs=nano.specs",
    "--specs=nosys.specs",
]

# ── source discovery ────────────────────────────────────────────────


def find_sources() -> list[Path]:
    """Find all .c / .s / .S source files, avoiding duplicates."""
    seen: set[str] = set()
    sources: list[Path] = []

    # Patterns: (directory, glob_pattern)
    patterns = [
        ("Core/Src", "*.c"),
        ("Core/Src", "*.S"),
        ("Core/Startup", "*.s"),
        ("Core/Startup", "*.S"),
        ("NetXDuo/App", "*.c"),
        ("AZURE_RTOS/App", "*.c"),
        ("Drivers/STM32U5xx_HAL_Driver/Src", "*.c"),
        ("Drivers/BSP/B-U585I-IOT02A", "*.c"),
        ("Drivers/BSP/Components/hts221", "*.c"),
        ("Drivers/BSP/Components/lps22hh", "*.c"),
        ("Drivers/BSP/Components/ov5640", "*.c"),
        ("Drivers/BSP/Components/veml3235", "*.c"),
        ("Drivers/BSP/Components/mx_wifi", "*.c"),
        ("Drivers/BSP/Components/mx_wifi/core", "*.c"),
        ("Drivers/BSP/Components/mx_wifi/io_pattern", "*.c"),
        ("Middlewares/ST/threadx/common/src", "*.c"),
        ("Middlewares/ST/threadx/ports/cortex_m33/gnu/src", "*.S"),
        ("Middlewares/ST/threadx/ports/cortex_m33/gnu/src", "*.s"),
        ("Middlewares/ST/netxduo/common/src", "*.c"),
        ("Middlewares/ST/netxduo/addons/mqtt", "*.c"),
        ("Middlewares/ST/netxduo/addons/dhcp", "*.c"),
        ("Middlewares/ST/netxduo/addons/dns", "*.c"),
        ("Middlewares/ST/netxduo/addons/sntp", "*.c"),
        ("Middlewares/ST/netxduo/common/drivers/wifi/mxchip", "*.c"),
    ]

    # Core/Src has project-specific overrides for some HAL files
    core_overrides = {
        "stm32u5xx_hal_msp.c",
        "stm32u5xx_hal_timebase_tim.c",
    }

    # Files that are not standalone sources (included by other sources)
    exclude_files = {
        "nx_driver_framework.c",  # included by nx_driver_emw3080.c
    }

    for dirname, glob_pat in patterns:
        d = PROJECT / dirname
        if not d.is_dir():
            continue
        for f in sorted(d.glob(glob_pat)):
            if f.name in core_overrides and str(f.parent) != str(PROJECT / "Core/Src"):
                continue  # use the Core/Src version instead
            if f.name in exclude_files:
                continue
            if f.name not in seen:
                seen.add(f.name)
                sources.append(f)

    return sources


def compile_file(src: Path, idx: int) -> Path | None:
    """Compile a single source file; return .o path or None on failure."""
    # Unique object name: use parent-dir prefix to avoid collisions
    parent = (
        src.parent.name
        if src.parent.name not in ("", "src", "Src")
        else src.parent.parent.name
    )
    obj_name = f"{parent}_{src.stem}.o"
    obj = BUILD_DIR / "obj" / obj_name

    obj.parent.mkdir(parents=True, exist_ok=True)

    suffix = src.suffix.lower()
    defines = [f"-D{d}" for d in DEFINES]
    incs = [f"-I{PROJECT / d}" for d in INCLUDES]
    if suffix in (".s",):
        # Assembly files also need defines (ThreadX .S files check TX_SINGLE_MODE_*)
        cmd = [str(CC)] + MCU_FLAGS + defines + incs + ["-c", str(src), "-o", str(obj)]
    else:
        cmd = [str(CC)] + COMMON_FLAGS + defines + incs
        cmd += ["-std=gnu11", "-c", str(src), "-o", str(obj)]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT))
    if result.returncode != 0:
        rel = src.relative_to(PROJECT)
        print(f"  ERROR [{idx}] {rel}")
        # Show last few lines of error
        lines = result.stderr.strip().splitlines()
        for line in lines[-10:]:
            print(f"    {line}")
        return None

    stderr = result.stderr.strip()
    if stderr:
        # Show only warnings, not info
        for line in stderr.splitlines():
            if "warning:" in line:
                rel = src.relative_to(PROJECT)
                print(f"  WARN [{idx}] {rel}: {line.strip()}")

    return obj


def main():
    config = PROJECT / "Core/Inc/network_config.h"
    if not config.is_file():
        sys.exit(
            "Copy Core/Inc/network_config.h.example to network_config.h and configure your network."
        )
    if not CC.is_file():
        sys.exit(
            "ARM compiler not found. Set ARM_TOOLCHAIN_DIR to the toolchain bin directory."
        )
    BUILD_DIR.mkdir(exist_ok=True)
    (BUILD_DIR / "obj").mkdir(exist_ok=True)

    # ── node name ───────────────────────────────────────────────────
    NODE_NAME = os.environ.get("NODE_NAME", "r2001/n1")
    node_define = f'NODE_NAME="{NODE_NAME}"'
    DEFINES.append(node_define)
    print(f"Building for NODE_NAME = {NODE_NAME}")

    sources = find_sources()
    print(f"Found {len(sources)} source files")

    objects = []
    failed = 0
    for i, src in enumerate(sources):
        obj = compile_file(src, i + 1)
        if obj:
            objects.append(obj)
        else:
            failed += 1
            if failed > 20:
                print(f"\nToo many failures ({failed}), aborting")
                sys.exit(1)

    if failed:
        print(f"\nFailed to compile {failed} files, aborting link")
        sys.exit(1)

    print(f"\nCompiled {len(objects)} objects successfully")

    # ── link ────────────────────────────────────────────────────────
    elf = BUILD_DIR / "IoTNetDuoCam.elf"
    ld_cmd = [str(CC)] + LINKER_FLAGS + [str(o) for o in objects] + ["-o", str(elf)]
    print(f"Linking {len(objects)} objects → {elf.name}...")
    result = subprocess.run(ld_cmd, capture_output=True, text=True, cwd=str(PROJECT))
    if result.returncode != 0:
        print("LINK ERROR:")
        lines = result.stderr.strip().splitlines()
        for line in lines[-40:]:
            print(line)
        sys.exit(1)
    # Print memory usage from link
    stderr = result.stderr.strip()
    if stderr:
        for line in stderr.splitlines():
            if any(
                kw in line for kw in ("text", "data", "bss", "FLASH", "RAM", "Memory")
            ):
                print(line)

    # ── bin / hex ───────────────────────────────────────────────────
    for fmt, ext in [("binary", ".bin"), ("ihex", ".hex")]:
        out = BUILD_DIR / f"IoTNetDuoCam{ext}"
        subprocess.run([str(OBJCOPY), "-O", fmt, str(elf), str(out)], check=True)
        print(f"Created {out} ({out.stat().st_size} bytes)")

    subprocess.run([str(SIZE), str(elf)])
    print(f"\nDone. Output in {BUILD_DIR}/")


if __name__ == "__main__":
    main()
