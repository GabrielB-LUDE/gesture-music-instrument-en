"""Gesture metrics: everything that reveals the pose of a hand.

The instrument should be playable while the hand stays relaxed in front
of the body. Nobody should have to walk across the image to get louder.
That is why this module reads only the *pose* of the hand and never its
position:

    finger_count   how many fingers are up (0..4)     -> density of the beat
    lean           tilt of the hand outwards          -> volume
    thumb_open     how far the thumb sticks out       -> timbre
    palm_facing    palm or back of the hand           -> hold (freeze)
    pinch          thumb and index finger touch       -> grab and place
    openness       fist to wide open hand             -> group macros
    grip_point     where the fingers grab             -> aiming in the columns

All values are normalised to the hand size. So it does not matter how
far someone stands from the camera.

On mirroring: `palm_facing` and `lean` factor out the handedness.
MediaPipe assigns the label as if the image were *not* mirrored.
If you mirror the image, label and geometry flip together - so the
product of both stays the same. Both values are therefore correct with
and without the selfie view.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from ..config import MappingConfig
from ..vision.types import (
    FINGER_CHAINS,
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_TIP,
    PINKY_MCP,
    THUMB_TIP,
    WRIST,
    HandObservation,
)

# Area of the palm triangle with the hand turned fully towards the camera,
# measured in hand sizes squared. Serves as the reference for palm_facing.
PALM_REFERENCE = 0.62


def clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def clamp_signed(value: float) -> float:
    return float(min(1.0, max(-1.0, value)))


def _distance(points: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(points[a][:2] - points[b][:2]))


def _hand_sign(obs: HandObservation) -> float:
    """+1 for a right hand in the unmirrored image, otherwise -1."""
    return 1.0 if obs.handedness.startswith("r") else -1.0


# ----------------------------------------------------------------------
# Fingers
# ----------------------------------------------------------------------
def finger_curl(obs: HandObservation, chain) -> float:
    """0 = finger extended, 1 = finger fully curled."""
    mcp, _pip, tip = chain
    base = _distance(obs.landmarks, WRIST, mcp)
    if base < 1e-6:
        return 0.0
    ratio = _distance(obs.landmarks, WRIST, tip) / base
    # Extended is about 2.4, a closed fist about 1.1
    return clamp01((2.35 - ratio) / 1.15)


def curl_values(obs: HandObservation) -> List[float]:
    return [finger_curl(obs, chain) for chain in FINGER_CHAINS]


def extended_fingers(obs: HandObservation, config: MappingConfig) -> List[bool]:
    """Index, middle, ring, little finger - each extended?"""
    return [curl <= config.finger_extend_max for curl in curl_values(obs)]


def finger_count(obs: HandObservation, config: MappingConfig) -> int:
    """Number of extended fingers without the thumb, 0..4.

    The thumb deliberately does not count: it controls the timbre on its
    own and should not distort the counting gesture.
    """
    return sum(1 for flag in extended_fingers(obs, config) if flag)


def is_fist(obs: HandObservation, config: MappingConfig) -> bool:
    return finger_count(obs, config) == 0


def density_from_count(count: int) -> float:
    """Map the finger count onto the four density levels of a preset."""
    level = min(3, max(0, count - 1))
    return (level + 0.5) / 4.0


# ----------------------------------------------------------------------
# Pose
# ----------------------------------------------------------------------
def tilt(obs: HandObservation) -> float:
    """Tilt in the image plane, -1 = fully to the left, +1 = fully to the right.

    0 means: the fingers point upwards. What is measured is the axis from
    the wrist to the base joint of the middle finger, i.e. the back of the
    hand itself - which stays stable even when the fingers are curled.
    """
    axis = obs.landmarks[MIDDLE_MCP][:2] - obs.landmarks[WRIST][:2]
    if float(np.linalg.norm(axis)) < 1e-6:
        return 0.0
    angle = math.atan2(float(axis[0]), -float(axis[1]))
    return clamp_signed(angle / (math.pi * 0.5))


def lean(obs: HandObservation, config: MappingConfig) -> float:
    """Outward tilt as 0..1. An upright hand sits in the middle.

    Tilting outwards (away from the body) makes it louder, inwards quieter.
    That way both hands feel the same.
    """
    span = max(config.lean_degrees, 1.0) / 90.0
    outward = -tilt(obs) * _hand_sign(obs)
    return clamp01(0.5 + 0.5 * (outward / span))


def palm_facing(obs: HandObservation) -> float:
    """+1 = palm towards the camera, 0 = hand on its edge, -1 = back.

    Computed from the signed area of the triangle
    wrist / index base joint / little finger base joint. When the
    wrist turns, the triangle flips over - that is exactly the gesture
    for freezing.
    """
    points = obs.landmarks
    a = points[INDEX_MCP][:2] - points[WRIST][:2]
    b = points[PINKY_MCP][:2] - points[WRIST][:2]
    cross = float(a[0] * b[1] - a[1] * b[0])
    area = -cross / (obs.scale * obs.scale * PALM_REFERENCE)
    return clamp_signed(area * _hand_sign(obs))


def thumb_open(obs: HandObservation, config: MappingConfig) -> float:
    """How far the thumb sticks out from the index base joint, 0..1."""
    raw = _distance(obs.landmarks, THUMB_TIP, INDEX_MCP) / obs.scale
    span = max(config.thumb_max - config.thumb_min, 1e-6)
    return clamp01((raw - config.thumb_min) / span)


def pinch_value(obs: HandObservation) -> float:
    """Distance from thumb tip to index tip in hand sizes."""
    return _distance(obs.landmarks, THUMB_TIP, INDEX_TIP) / obs.scale


def grip_point(obs: HandObservation, config: MappingConfig) -> np.ndarray:
    """The point where thumb and index finger meet, shape (2,).

    This is what you aim with in the two columns. Otherwise the tracked
    point is the centre of the hand between wrist and finger base joints -
    which sits roughly one slot height below the fingers you visibly grab
    with. Exactly this offset made the selection miss:
    the hand pointed at one slot, but the one below it was selected.

    The point is computed from the hand axis and not from the
    fingertips. When grabbing, the index finger curls in; an aiming
    point at the tips would move exactly at the moment when the
    selection has to stand still. The axis wrist -> base joint
    of the middle finger, on the other hand, is unaffected by how the
    fingers are currently positioned, and turns with the hand.
    """
    points = obs.landmarks
    wrist = points[WRIST][:2]
    axis = points[MIDDLE_MCP][:2] - wrist
    return np.asarray(wrist + axis * config.grip_reach, dtype=np.float32)


def is_pinch(obs: HandObservation, config: MappingConfig) -> bool:
    """Thumb and index finger touch - and nothing else.

    The second condition separates the pinch from the fist: there the
    thumb lies over all fingers and is just as far from the middle finger
    as from the index finger. In a real pinch the middle finger is clearly
    further away.
    """
    index = pinch_value(obs)
    if index > config.pinch_max:
        return False
    middle = _distance(obs.landmarks, THUMB_TIP, MIDDLE_TIP) / obs.scale
    return middle > index * config.pinch_clearance


# ----------------------------------------------------------------------
# Overall pose - basis of the group macros
# ----------------------------------------------------------------------
def openness(obs: HandObservation, config: MappingConfig) -> float:
    """0 = closed fist, 1 = all fingers extended, thumb spread.

    This makes the group macros pure pose gestures: opening the hands
    together pulls the riser up, clenching the fists together
    triggers the drop. Nobody has to raise their hands above their head
    for this - the arm can stay relaxed in front of the body.

    The finger count carries three quarters, the thumb one quarter. That
    way a hand with four extended fingers alone is not yet enough for
    the riser: the thumb has to be deliberately spread as well.
    """
    fingers = finger_count(obs, config) / 4.0
    return clamp01(fingers * 0.75 + thumb_open(obs, config) * 0.25)
