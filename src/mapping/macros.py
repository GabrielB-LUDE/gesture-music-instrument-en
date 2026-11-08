"""Group macros: shared build-up and drop.

Both work through the pose of the hands, not their position.
The hands may stay in front of the body:

    Riser  all hands open wide together (fingers extended,
           thumb spread). The value rises slowly and can be
           used as a filter or send automation.
    Drop   right after a full riser everyone clenches their fist
           together. Then a short pulse is fired.

What is measured is the median hand openness, so that a single hand
that is doing something else neither triggers nor blocks the macro.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import List

from ..config import MappingConfig
from ..vision.types import MacroState


@dataclass
class MacroDetector:
    config: MappingConfig
    riser: float = 0.0
    drop: float = 0.0
    _open_frames: int = 0
    _closed_frames: int = 0
    _armed: bool = False

    def update(self, openness: List[float]) -> MacroState:
        active = len(openness)
        median = statistics.median(openness) if openness else 0.0

        if active >= 2 and median >= self.config.riser_open:
            self._open_frames += 1
            self._closed_frames = 0
        elif active >= 2 and median <= self.config.drop_closed:
            self._closed_frames += 1
            self._open_frames = 0
        else:
            self._open_frames = max(0, self._open_frames - 1)
            self._closed_frames = 0

        if self._open_frames >= self.config.riser_frames:
            self.riser = min(1.0, self.riser + 0.035)
            if self.riser > 0.75:
                self._armed = True
        else:
            self.riser = max(0.0, self.riser - 0.05)

        self.drop = max(0.0, self.drop - 0.08)
        if self._armed and self._closed_frames >= self.config.drop_frames:
            self.drop = 1.0
            self.riser = 0.0
            self._armed = False
            self._open_frames = 0

        return MacroState(
            riser=round(self.riser, 4),
            drop=round(self.drop, 4),
            group_open=round(median, 4),
            active_hands=active,
        )
