"""Central configuration.

All values can be overridden through a JSON file (config.json).
Usage: Config.load("config.json")
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict, fields
from typing import Any, Dict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

HAND_MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

MAX_PERFORMERS = 3


@dataclass
class CameraConfig:
    device: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 60
    mirror: bool = True  # selfie view, image is mirrored horizontally


@dataclass
class TrackingConfig:
    max_hands: int = 6
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    # How many frames a hand may be missing before its channel goes silent.
    hold_frames: int = 8


@dataclass
class MappingConfig:
    """How a hand pose is translated into numbers.

    Deliberately without any image position: you play with the pose of the
    hand, not with its location. The hand can stay relaxed in front of the
    body.
    """

    smoothing: float = 0.35        # 0 = no smoothing, 1 = frozen

    # count fingers -> density of the beat
    finger_extend_max: float = 0.45   # up to here a finger counts as extended
    finger_hold_frames: int = 3       # a count must be stable for this long

    # tilt the hand -> volume
    lean_degrees: float = 55.0     # tilt this far for the full range

    # spread the thumb -> timbre
    thumb_min: float = 0.33        # thumb against the hand, in hand sizes
    thumb_max: float = 0.88        # thumb spread wide

    # turn the wrist -> hold the loop
    flip_threshold: float = 0.25   # dead zone around the edge of the hand
    flip_frames: int = 3           # the side must be right for this many frames
    flip_arm: float = 0.55         # above this the hand counts as flat to the camera

    # Where the hand points: the point where thumb and index finger
    # meet, measured in hand sizes above the wrist along the
    # hand axis. This is what aims into the columns.
    grip_reach: float = 1.30

    # thumb and index finger -> grab and place
    pinch_max: float = 0.30        # closed, in hand sizes
    pinch_clearance: float = 1.4   # middle finger must be clearly further away
    # The tight threshold separates the pinch from the fist: there the
    # thumb may happen to lie close to the index finger depending on the pose.
    # Since the pinch also places loops, such a mix-up would be costly.

    # Macro trigger, measured on the median hand openness (0 = fist)
    riser_open: float = 0.85       # all hands must be open at least this far
    riser_frames: int = 12
    drop_closed: float = 0.15      # all fists must be closed at least this far
    drop_frames: int = 6


@dataclass
class ForgeConfig:
    """The two side columns, loops in the hand, fusion and placing.

    On the left the library with the finished beats, on the right the rack
    with the self-built loops. Both columns have fixed slots: a slot
    stays in its place, no matter how much is in it at the moment. As with
    the trash bin, it is deliberately the position of the hand that decides
    in the columns, not its pose - you see the slot and reach for it.
    """

    # library on the left
    library_x: float = 0.115        # centre of the column, normalised
    library_width: float = 0.185    # width of a slot
    library_top: float = 0.205
    library_bottom: float = 0.925
    library_capture_x: float = 0.32 # up to here an empty hand counts as "at the column"

    # rack on the right
    rack_x: float = 0.885
    rack_width: float = 0.185
    rack_top: float = 0.205
    rack_bottom: float = 0.775
    rack_capture_x: float = 0.68    # from here an empty hand counts as "at the rack"
    max_parked: int = 5             # number of slots in the rack

    # Selection in a column. Smoothing keeps a resting hand
    # still, but must not slow down a moving one - otherwise the
    # column still points at a slot the fingers have long passed.
    # That is why both are released as soon as the hand visibly moves.
    select_hysteresis: float = 0.22 # extra distance in slot heights before switching
    select_smoothing: float = 0.35  # smoothing of the resting hand height, 0 = raw
    select_settle: float = 0.012    # distance per frame above which smoothing and lock are dropped

    take_frames: int = 3           # hold the pinch this long, then the hand grabs
    take_cooldown: int = 16        # briefly locked afterwards

    # Holding
    release_frames: int = 10       # hand gone: loop moves to the rack

    # Freezing by turning the wrist
    freeze_frames: int = 3         # the back of the hand must be visible this long

    # Fusion: bring hands together
    fuse_distance: float = 0.22    # hands must come this close
    fuse_frames: int = 3           # hold together this long
    fuse_duration: int = 14        # length of the spiral animation in frames
    max_recipes: int = 4           # this many beats fit into one loop

    # Placing by a pinch with the loop in the hand
    min_hold_frames: int = 10      # nothing can be placed before this
    place_frames: int = 4          # hold the pinch this long, then release
    discard_frames: int = 22       # holding the pinch this long throws the loop away
    fly_frames: int = 12           # flight to the rack
    max_tokens: int = 8

    # Trash bin below the rack: carrying a held loop there and
    # holding briefly deletes it. Together with the two columns, the only
    # place where the position of the hand matters - playing itself
    # stays completely position-free.
    trash_x: float = 0.885
    trash_y: float = 0.855
    trash_radius: float = 0.075
    trash_frames: int = 24


@dataclass
class TrackConfig:
    """Re-identification of individual hands across several frames."""

    max_match_distance: float = 0.16
    max_missing_frames: int = 12
    velocity_damping: float = 0.55


@dataclass
class OscConfig:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 9000
    send_rate: int = 60            # maximum messages per second
    # Only send when a value has changed by more than delta.
    min_delta: float = 0.004


@dataclass
class AudioConfig:
    enabled: bool = True           # internal sound engine (no DAW needed)
    # Where the beats come from:
    # "auto"    own loops from the loops/ folder, otherwise the built-in ones
    # "loops"   own loops only
    # "builtin" built-in beats only
    library: str = "auto"
    samplerate: int = 44100
    blocksize: int = 256
    bpm: float = 128.0
    master_gain: float = 0.80
    device: Any = None             # None = system default


@dataclass
class UiConfig:
    show_window: bool = True
    fullscreen: bool = False
    window_name: str = "Gesture Music Instrument"
    draw_landmarks: bool = True
    countdown_seconds: float = 3.0


@dataclass
class Config:
    performers: int = 1
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    forge: ForgeConfig = field(default_factory=ForgeConfig)
    tracks: TrackConfig = field(default_factory=TrackConfig)
    osc: OscConfig = field(default_factory=OscConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    ui: UiConfig = field(default_factory=UiConfig)

    @property
    def hand_budget(self) -> int:
        """Two hands per person."""
        return min(max(self.performers, 1), MAX_PERFORMERS) * 2

    @classmethod
    def load(cls, path: str | None = None) -> "Config":
        cfg = cls()
        if not path:
            return cfg
        if not os.path.exists(path):
            return cfg
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cfg.merge(data)

    def merge(self, data: Dict[str, Any]) -> "Config":
        sub = {f.name: f.type for f in fields(self)}
        for key, value in data.items():
            if key not in sub:
                continue
            current = getattr(self, key)
            if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if hasattr(current, sub_key):
                        setattr(current, sub_key, sub_value)
            else:
                setattr(self, key, value)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
