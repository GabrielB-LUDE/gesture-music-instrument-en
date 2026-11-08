"""Geometry of the two side columns: one source for the forge and the display.

On the left is the library, on the right the rack. Both are columns of
equally tall slots in fixed places - no scroll menu, no list that
shifts as soon as something is added. That was exactly the problem
before: the placed loops crowded into the same column, everything got
smaller and shifted with every placement.

So that a hand hits exactly the slot the display draws, the forge
(selection) and the display (drawing) share the same formula - otherwise
selection and image drift apart.

`Column.pick` is the second part of the answer to "selecting is tedious":
the hand must move past the centre of the neighbouring slot before the
selection switches. Without this lock the selection flickers at every slot
boundary, and that is exactly where the hand rests most often. The lock is
only meant for the resting hand, though - imposing it on a moving hand
too makes the selection lag behind. How strongly it applies is therefore
decided by the caller (see `TokenForge._update_browser`).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Column:
    """A column of `slots` equally tall slots, normalised 0..1."""

    x: float
    top: float
    bottom: float
    slots: int

    @classmethod
    def packed(cls, x: float, top: float, bottom: float, slots: int,
               max_pitch: float = 0.16) -> "Column":
        """Column that does not fall apart when there are few slots.

        Eight beats fill the column on their own. With two own loops a slot
        would otherwise be a third of the screen tall - big enough
        to hit, but it looks as if something is missing. So in that case
        the column becomes shorter and moves to the centre.
        """
        slots = max(slots, 1)
        span = min(bottom - top, max_pitch * slots)
        centre = (top + bottom) * 0.5
        return cls(x, centre - span * 0.5, centre + span * 0.5, slots)

    @property
    def span(self) -> float:
        return max(self.bottom - self.top, 1e-6)

    @property
    def pitch(self) -> float:
        """Distance between two slot centres."""
        return self.span / max(self.slots, 1)

    def slot_y(self, index: int) -> float:
        """Centre of the `index`-th slot."""
        return self.top + self.pitch * (index + 0.5)

    def position(self, y: float) -> float:
        """Continuous slot position of a hand at height `y`."""
        return (y - self.top) / self.pitch - 0.5

    def nearest(self, y: float) -> int:
        if self.slots <= 1:
            return 0
        return max(0, min(self.slots - 1, int(round(self.position(y)))))

    def pick(self, y: float, current: int, hysteresis: float = 0.35) -> int:
        """Selection with a lock: it only switches once clearly inside the neighbouring slot.

        `current` is the previous selection, `hysteresis` the extra
        distance in slot heights the hand must travel beyond that.
        """
        if self.slots <= 1:
            return 0
        current = max(0, min(self.slots - 1, current))
        position = self.position(y)
        if position > current + 0.5 + hysteresis:
            return min(self.slots - 1, int(round(position)))
        if position < current - 0.5 - hysteresis:
            return max(0, int(round(position)))
        return current


def library_slot_y(index: int, count: int, top: float, bottom: float) -> float:
    """Normalised Y position of the `index`-th of `count` slots."""
    return Column(0.0, top, bottom, count).slot_y(index)


def nearest_library_index(y: float, count: int, top: float, bottom: float) -> int:
    """Which slot is closest to a hand at height `y`."""
    if count <= 0:
        return 0
    return Column(0.0, top, bottom, count).nearest(y)
