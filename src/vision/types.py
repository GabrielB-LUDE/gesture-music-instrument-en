"""Data types shared between vision, mapping and UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:  # only for the type annotation, otherwise there would be an import cycle
    from ..mapping.tokens import Token

import numpy as np

# Landmark indices of the MediaPipe hand (21 points)
WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

FINGER_CHAINS = (
    (INDEX_MCP, INDEX_PIP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    (RING_MCP, RING_PIP, RING_TIP),
    (PINKY_MCP, PINKY_PIP, PINKY_TIP),
)

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


@dataclass
class HandObservation:
    """A detected hand in normalised image coordinates (0..1)."""

    landmarks: np.ndarray          # shape (21, 3)
    handedness: str                # "left" or "right"
    score: float = 1.0

    @property
    def wrist(self) -> np.ndarray:
        return self.landmarks[WRIST]

    @property
    def center(self) -> np.ndarray:
        """Centre from the wrist and the middle finger base joint."""
        return (self.landmarks[WRIST] + self.landmarks[MIDDLE_MCP]) * 0.5

    @property
    def scale(self) -> float:
        """Hand size as the reference for distance-based metrics."""
        diff = self.landmarks[MIDDLE_MCP][:2] - self.landmarks[WRIST][:2]
        return float(max(np.linalg.norm(diff), 1e-6))


@dataclass
class Frame:
    """A camera frame plus metadata."""

    image: np.ndarray
    index: int = 0
    timestamp_ms: int = 0

    @property
    def size(self):
        h, w = self.image.shape[:2]
        return w, h


@dataclass
class SoundSlot:
    """A sounding layer: a preset with its values.

    A token with one beat yields one slot, a fused token
    yields one slot per contained beat. The sound engine keeps
    its own voice for every slot.
    """

    key: str                       # stable across frames: "<token>:<index>"
    preset_id: str
    gain: float = 0.0
    cutoff: float = 0.5
    density: float = 0.0


@dataclass
class BrowserEntry:
    """A slot in one of the two columns.

    "preset" is a fresh beat from the library on the left, "loop" an
    already built loop that sits on the rack on the right and can go back
    into the hand.
    """

    kind: str
    preset_ids: List[str]
    token_id: Optional[int] = None


@dataclass
class HandView:
    """What the display needs to know about a hand.

    The values come ready-prepared from the forge, so that the
    display does not have to compute the gesture recognition a second time.
    """

    hand_id: int
    x: float
    y: float
    # Where the hand points: the point where thumb and index finger
    # grab. The columns select based on this, and the pointer line starts
    # from here - otherwise the image would point somewhere other than the selection.
    aim_x: float = 0.5
    aim_y: float = 0.5
    handedness: str = "right"
    holding: bool = False
    token_id: Optional[int] = None
    zone: str = ""                 # "library", "rack" or empty
    selection: int = 0             # slot the hand is currently pointing at
    take_progress: float = 0.0
    delete_progress: float = 0.0   # rack only: progress towards deletion
    fingers: int = 0
    facing: float = 1.0
    lean: float = 0.5
    thumb: float = 0.0
    landmarks: Optional[np.ndarray] = None


@dataclass
class MacroState:
    riser: float = 0.0
    drop: float = 0.0
    group_open: float = 0.0   # median hand openness, 0 = fists, 1 = open
    active_hands: int = 0


@dataclass
class EngineSnapshot:
    """Everything OSC, audio and display need per frame."""

    tokens: List["Token"]
    slots: List[SoundSlot]
    library: List[BrowserEntry]              # left column, one slot per beat
    rack: List[Optional[BrowserEntry]]       # right column, empty slots as None
    hands: List[HandView]
    macros: MacroState
    fps: float = 0.0
