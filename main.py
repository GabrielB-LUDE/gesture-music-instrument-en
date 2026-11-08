#!/usr/bin/env python3
"""Entry point of the instrument.

Examples:
    python main.py                      # camera, 3 players, OSC + internal engine
    python main.py --performers 1       # solo mode for practising
    python main.py --mock --no-window   # test run without a camera
"""

import os
import sys

# Quieten the log noise from OpenCV and MediaPipe
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from src.app import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
