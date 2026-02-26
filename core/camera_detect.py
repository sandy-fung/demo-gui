"""Auto-detect DVS (FX3) and RGB cameras via v4l2-ctl."""

import subprocess
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class CameraDetectResult:
    dvs_device: Optional[str] = None   # e.g. "/dev/video0"
    rgb_device: Optional[str] = None   # e.g. "/dev/video2"
    dvs_name: Optional[str] = None     # e.g. "FX3: FX3 (usb-...)"
    rgb_name: Optional[str] = None     # e.g. "Live Streamer CAM 310P: ..."


# Device name prefixes to skip (not user cameras)
_SKIP_PREFIXES = ("NVIDIA Tegra",)


def _parse_v4l2_sections(output: str) -> list[tuple[str, list[str]]]:
    """Parse v4l2-ctl --list-devices output into (header, [device_paths]) pairs."""
    sections: list[tuple[str, list[str]]] = []
    current_header = None
    current_devices: list[str] = []

    for line in output.strip().split("\n"):
        if not line:
            continue
        # Indented lines are device paths; non-indented lines are headers
        if line[0].isspace():
            match = re.search(r"(/dev/video\d+)", line.strip())
            if match:
                current_devices.append(match.group(1))
        else:
            # Save previous section
            if current_header is not None:
                sections.append((current_header, current_devices))
            current_header = line.rstrip(":")
            current_devices = []

    # Save last section
    if current_header is not None:
        sections.append((current_header, current_devices))

    return sections


def detect_cameras() -> CameraDetectResult:
    """Auto-detect DVS (FX3) and RGB cameras via v4l2-ctl.

    - DVS: section header containing "FX3: FX3" → first /dev/video*
    - RGB: first non-FX3, non-Tegra section with /dev/video*
    - Returns None for devices not found
    """
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True, text=True, timeout=5,
        )
        output = result.stdout
    except subprocess.TimeoutExpired:
        print("[camera_detect] v4l2-ctl timed out")
        return CameraDetectResult()
    except FileNotFoundError:
        print("[camera_detect] v4l2-ctl not found (install v4l-utils)")
        return CameraDetectResult()
    except Exception as e:
        print(f"[camera_detect] error: {e}")
        return CameraDetectResult()

    sections = _parse_v4l2_sections(output)
    cam = CameraDetectResult()

    for header, devices in sections:
        if not devices:
            continue

        if "FX3: FX3" in header:
            cam.dvs_device = devices[0]
            cam.dvs_name = header
        elif not any(header.startswith(p) for p in _SKIP_PREFIXES):
            # First non-skipped, non-FX3 section → RGB
            if cam.rgb_device is None:
                cam.rgb_device = devices[0]
                cam.rgb_name = header

    return cam
