"""Core logic.

Flow per frame:

    detect hands
      -> keep identities across frames (HandTracker)
      -> grab, shape, hold, fuse, place loops (TokenForge)
      -> snapshot to OSC, audio and display

There are no fixed channels any more. What sounds follows from the
tokens that are currently held in a hand or placed in the image.
"""

from __future__ import annotations

import time
from typing import List, Optional

from .config import Config
from .mapping.macros import MacroDetector
from .mapping.tokens import TokenForge
from .vision.hand_tracker import HandTracker
from .vision.types import EngineSnapshot, HandObservation


class InstrumentEngine:
    def __init__(self, config: Config, seed: Optional[int] = None):
        self.config = config
        self.hands = HandTracker(config.tracks)
        self.forge = TokenForge(config.forge, config.mapping, seed=seed)
        self.macro = MacroDetector(config.mapping)

        self._last_time = time.perf_counter()
        self._fps = 0.0

    # ------------------------------------------------------------------
    def set_performers(self, performers: int) -> None:
        self.config.performers = performers
        self.hands.reset()
        self.forge.clear()

    def shuffle_library(self) -> None:
        self.forge.shuffle_library()

    def place_all(self) -> None:
        self.forge.place_all()

    def clear(self) -> None:
        self.forge.clear()

    # ------------------------------------------------------------------
    def update(
        self, observations: List[HandObservation], dt: Optional[float] = None
    ) -> EngineSnapshot:
        frame_time = self._tick() if dt is None else dt
        tracks = self.hands.update(observations)
        tokens = self.forge.update(tracks, frame_time)
        macros = self.macro.update(self.forge.live_openness)
        # The detector only sees the hands that are currently tuning a loop.
        # But the display should show how many hands are there at all.
        macros.active_hands = len(tracks)
        return EngineSnapshot(
            tokens=tokens,
            slots=self.forge.slots(),
            library=self.forge.library_entries(),
            rack=self.forge.rack_entries(),
            hands=self.forge.hands,
            macros=macros,
            fps=round(self._fps, 1),
        )

    # ------------------------------------------------------------------
    def _tick(self) -> float:
        """Frame timing. Returns the frame time in seconds.

        The limits catch the first frame and short hiccups, otherwise
        the swipe detection would overreact.
        """
        now = time.perf_counter()
        delta = min(max(now - self._last_time, 1.0 / 120.0), 1.0 / 10.0)
        self._last_time = now
        instant = 1.0 / delta
        self._fps = instant if self._fps == 0 else self._fps * 0.9 + instant * 0.1
        return delta
