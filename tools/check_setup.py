#!/usr/bin/env python3
"""Checks whether everything is available on this machine.

Usage:  python tools/check_setup.py
"""

from __future__ import annotations

import os
import platform
import sys

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("GLOG_minloglevel", "2")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OK = "  ok   "
WARN = " warn  "
FAIL = "missing"


def line(state: str, text: str) -> None:
    print(f"[{state}] {text}")


def check_python() -> bool:
    version = sys.version_info
    good = (3, 9) <= (version.major, version.minor) < (3, 14)
    line(OK if good else WARN, f"Python {platform.python_version()} ({platform.system()})")
    return good


def check_package(name: str, import_name: str | None = None) -> bool:
    try:
        module = __import__(import_name or name)
        version = getattr(module, "__version__", "?")
        line(OK, f"{name} {version}")
        return True
    except Exception as exc:
        line(FAIL, f"{name} ({exc})")
        return False


def check_model() -> bool:
    from src.config import HAND_MODEL_PATH

    if os.path.exists(HAND_MODEL_PATH) and os.path.getsize(HAND_MODEL_PATH) > 100_000:
        size = os.path.getsize(HAND_MODEL_PATH) / 1_000_000
        line(OK, f"Hand model present ({size:.1f} MB)")
        return True
    line(WARN, "Hand model missing, will be downloaded on first start")
    return False


def check_camera(index: int = 0) -> bool:
    try:
        import cv2
    except Exception:
        line(FAIL, "OpenCV missing, camera cannot be checked")
        return False

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        line(FAIL, f"Camera {index} not reachable")
        return False
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        line(FAIL, f"Camera {index} delivers no image")
        return False
    line(OK, f"Camera {index} delivers {frame.shape[1]}x{frame.shape[0]}")
    return True


def check_audio() -> bool:
    try:
        import sounddevice as sd
    except Exception as exc:
        line(WARN, f"sounddevice not usable ({exc}). Internal engine stays off.")
        return False
    try:
        device = sd.query_devices(kind="output")
        line(OK, f"Audio output: {device['name']}")
        return True
    except Exception as exc:
        line(WARN, f"no audio output found ({exc})")
        return False


def main() -> int:
    print("System check Gesture Music Instrument\n")
    results = [
        check_python(),
        check_package("numpy"),
        check_package("opencv-python", "cv2"),
        check_package("mediapipe"),
        check_package("python-osc", "pythonosc"),
    ]
    check_audio()
    check_model()
    check_camera(int(sys.argv[1]) if len(sys.argv) > 1 else 0)

    print()
    if all(results):
        print("Everything important is there. Start with:  python main.py")
        return 0
    print("Something is missing. Please run first:  pip install -r requirements.txt")
    return 1


if __name__ == "__main__":
    sys.exit(main())
