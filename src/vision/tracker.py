"""Hand tracking based on the MediaPipe Tasks API.

The HandLandmarker is used deliberately and not the old
`mediapipe.solutions` API: that was removed in MediaPipe 0.10.3x.
The HandLandmarker detects up to six hands at the same time, exactly
what is needed for three people.
"""

from __future__ import annotations

import math
import os
import urllib.request
from typing import List, Protocol

import cv2
import numpy as np

from ..config import HAND_MODEL_PATH, HAND_MODEL_URL, TrackingConfig
from .types import Frame, HandObservation


class TrackerError(RuntimeError):
    pass


def ensure_model(path: str = HAND_MODEL_PATH, url: str = HAND_MODEL_URL) -> str:
    """Downloads the hand landmarker model on the first start."""
    if os.path.exists(path) and os.path.getsize(path) > 100_000:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[tracker] Downloading model: {url}")
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as exc:  # pragma: no cover - depends on the network
        raise TrackerError(
            "Model could not be loaded. Please download it manually "
            f"({url}) and place it at models/hand_landmarker.task. Error: {exc}"
        ) from exc
    print(f"[tracker] Model saved: {path}")
    return path


class Tracker(Protocol):
    def process(self, frame: Frame) -> List[HandObservation]: ...
    def close(self) -> None: ...


class MediaPipeHandTracker:
    def __init__(self, config: TrackingConfig, model_path: str = HAND_MODEL_PATH):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as exc:  # pragma: no cover
            raise TrackerError(
                "mediapipe is not installed. `pip install -r requirements.txt`"
            ) from exc

        self._mp = mp
        self._config = config
        ensure_model(model_path)

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=config.max_hands,
            min_hand_detection_confidence=config.min_detection_confidence,
            min_hand_presence_confidence=config.min_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._last_timestamp = -1

    def process(self, frame: Frame) -> List[HandObservation]:
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        # MediaPipe requires strictly increasing timestamps.
        timestamp = max(frame.timestamp_ms, self._last_timestamp + 1)
        self._last_timestamp = timestamp

        result = self._landmarker.detect_for_video(image, timestamp)
        observations: List[HandObservation] = []
        for idx, landmarks in enumerate(result.hand_landmarks):
            points = np.array(
                [[lm.x, lm.y, getattr(lm, "z", 0.0)] for lm in landmarks],
                dtype=np.float32,
            )
            label, score = "right", 1.0
            if idx < len(result.handedness) and result.handedness[idx]:
                category = result.handedness[idx][0]
                label = str(category.category_name).lower()
                score = float(category.score)
            observations.append(
                HandObservation(landmarks=points, handedness=label, score=score)
            )
        return observations

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Mock tracker: allows a complete test run without a camera and without a model.
# --------------------------------------------------------------------------

# Hand model in hand units. u runs across the palm towards the
# thumb, v runs from the wrist to the fingers. One unit
# is the distance wrist -> base joint of the middle finger, exactly the
# measure that gestures.py uses as well.
PALM = {
    0: (0.00, 0.00),    # wrist
    5: (0.36, 0.95),    # index finger
    9: (0.00, 1.00),    # middle finger
    13: (-0.22, 0.95),  # ring finger
    17: (-0.45, 0.84),  # little finger
}
FINGERS = (
    # mcp, pip, dip, tip, finger length, fan angle in degrees
    (5, 6, 7, 8, 1.25, 11.0),
    (9, 10, 11, 12, 1.35, 2.0),
    (13, 14, 15, 16, 1.22, -11.0),
    (17, 18, 19, 20, 0.98, -22.0),
)
# Thumb as two end poses, interpolated in between: tucked across
# the palm, spread wide to the side.
THUMB_CHAIN = (1, 2, 3, 4)
THUMB_CLOSED = ((0.20, 0.12), (0.30, 0.36), (0.26, 0.55), (0.19, 0.68))
THUMB_OPEN = ((0.30, 0.10), (0.58, 0.28), (0.80, 0.30), (0.99, 0.32))
JOINT_FRACTIONS = (0.45, 0.73, 1.00)   # PIP, DIP, TIP along the finger


def synthetic_hand(
    center_x: float,
    center_y: float,
    handedness: str = "right",
    spread: float = 1.0,
    fist: bool = False,
    point: bool = False,
    scale: float = 0.12,
    fingers: int | None = None,
    thumb: float | None = None,
    tilt: float = 0.0,
    roll: float = 1.0,
    pinch: bool = False,
) -> HandObservation:
    """Creates a plausible 21-point hand for tests and demos.

    fingers: how many fingers are up (0..4), counted from the index finger.
    thumb:   0 = thumb tucked, 1 = spread wide.
    tilt:    tilt in degrees, 0 = fingers up, positive to the right.
    roll:    rotation of the wrist. +1 = palm towards the camera,
             0 = hand on its edge, -1 = back of the hand towards the camera.
    pinch:   thumb and index finger touch.
    spread:  how far the fingers are fanned out.
    fist / point are shorthands for fingers=0 and fingers=1.
    """
    if fingers is None:
        fingers = 0 if fist else (1 if point else 4)
    if thumb is None:
        thumb = 0.1 if fist else (0.5 if pinch else 1.0)
    fingers = int(min(4, max(0, fingers)))

    local = dict(PALM)

    # Blend the thumb between tucked and spread.
    amount = min(max(thumb, 0.0), 1.0)
    for index, closed, opened in zip(THUMB_CHAIN, THUMB_CLOSED, THUMB_OPEN):
        local[index] = (
            closed[0] + (opened[0] - closed[0]) * amount,
            closed[1] + (opened[1] - closed[1]) * amount,
        )

    fan = 0.25 + 0.75 * min(max(spread, 0.0), 1.0)
    for order, (mcp, pip, dip, tip, length, angle) in enumerate(FINGERS):
        base = local[mcp]
        theta = math.radians(angle * fan)
        direction = (math.sin(theta), math.cos(theta))
        extended = order < fingers
        if extended:
            for joint, fraction in zip((pip, dip, tip), JOINT_FRACTIONS):
                reach = length * fraction
                local[joint] = (base[0] + direction[0] * reach,
                                base[1] + direction[1] * reach)
        else:
            # Curled: extended up to the middle joint, then back
            # towards the palm.
            knuckle = (base[0] + direction[0] * length * 0.45,
                       base[1] + direction[1] * length * 0.45)
            folded = math.radians(angle * fan + 150.0)
            back = (math.sin(folded), math.cos(folded))
            local[pip] = knuckle
            for joint, fraction in ((dip, 0.28), (tip, 0.55)):
                local[joint] = (knuckle[0] + back[0] * length * fraction,
                                knuckle[1] + back[1] * length * fraction)

    if pinch:
        # Index finger bends towards the thumb tip.
        tip_target = local[4]
        local[8] = (tip_target[0] + 0.02, tip_target[1] + 0.02)
        local[7] = ((local[6][0] + local[8][0]) * 0.5,
                    (local[6][1] + local[8][1]) * 0.5)

    angle = math.radians(tilt)
    v_axis = (math.sin(angle), -math.cos(angle))
    u_axis = (math.cos(angle), math.sin(angle))
    hand_sign = 1.0 if handedness.startswith("r") else -1.0

    pts = np.zeros((21, 3), dtype=np.float32)
    for index in range(21):
        u, v = local[index]
        across = u * hand_sign * roll
        pts[index] = (
            center_x + scale * (u_axis[0] * across + v_axis[0] * v),
            center_y + scale * (u_axis[1] * across + v_axis[1] * v),
            0.0,
        )
    return HandObservation(landmarks=pts, handedness=handedness, score=0.95)


class MockTracker:
    """Virtual hands that stay in front of the body and play through the poses."""

    def __init__(self, config: TrackingConfig, performers: int = 3):
        self.config = config
        self.performers = performers

    def process(self, frame: Frame) -> List[HandObservation]:
        t = frame.index / 30.0
        observations: List[HandObservation] = []
        count = max(1, self.performers) * 2
        for index in range(count):
            phase = t * (0.55 + 0.09 * index) + index * 2.3
            lane = (index + 0.5) / count
            # Small residual movement, but the hands stay in place.
            x = 0.12 + 0.76 * lane + 0.03 * math.sin(phase)
            y = 0.45 + 0.05 * math.sin(phase * 0.83 + index)
            hand = "left" if index % 2 else "right"
            fingers = int(2.5 + 1.6 * math.sin(phase * 0.41 + index))
            thumb = 0.5 + 0.5 * math.sin(phase * 0.7)
            tilt = 38.0 * math.sin(phase * 0.55 + index)
            roll = 1.0 if math.sin(phase * 0.19 + index) > -0.4 else -1.0
            pinch = math.sin(phase * 0.23 + index * 1.7) > 0.93
            observations.append(
                synthetic_hand(
                    x, y, hand,
                    fingers=max(0, min(4, fingers)),
                    thumb=thumb, tilt=tilt, roll=roll, pinch=pinch,
                )
            )
        return observations

    def close(self) -> None:
        return None


def build_tracker(
    config: TrackingConfig,
    mock: bool = False,
    performers: int = 3,
    model_path: str = HAND_MODEL_PATH,
) -> Tracker:
    if mock:
        return MockTracker(config, performers=performers)
    return MediaPipeHandTracker(config, model_path=model_path)
