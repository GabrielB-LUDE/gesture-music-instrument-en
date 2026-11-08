"""Start screen.

One player, one button: "START THE PARTY". It can be triggered with the
keyboard (Enter or space) or without a keyboard, by holding the hand over
the button until the ring is full. That matters at a party,
because nobody should have to walk over to the laptop.

It is drawn with the same building blocks as the game display - same
type, same colours, same glass. The start screen is the
first impression; if it looks like a different program, the whole
instrument seems cheaper than it is.
"""

from __future__ import annotations

import math
from typing import List, Optional

import cv2
import numpy as np

from ..config import Config
from ..vision.types import HandObservation
from .theme import (
    ACCENT,
    ACCENT_DIM,
    INK,
    INK_FAINT,
    INK_SOFT,
    LINE_SOFT,
    PANEL,
    TRACK_HERO,
    TYPE_HERO,
    TYPE_LABEL,
    TYPE_MICRO,
    TYPE_TITLE,
    Painter,
    arc,
    backdrop,
    clamp01,
    label,
    label_width,
    mix,
    scale,
    text,
    text_width,
    unit,
)

GESTURES = (
    ("Pinch", "Take loop"),
    ("Tilt hand", "browse and louder"),
    ("Show fingers", "Thicken beat"),
    ("Turn wrist", "Hold loop"),
    ("Hands together", "fuse"),
)


class Lobby:
    """A single start button instead of a player-count selection."""

    def __init__(self, config: Config, dwell_frames: int = 34):
        self.config = config
        self.dwell_frames = dwell_frames
        self.progress = 0.0
        self._pulse = 0.0
        self.paint = Painter()
        self.u = 1.0
        # Button centred, normalised coordinates.
        self.rect = (0.30, 0.45, 0.40, 0.145)

    # ------------------------------------------------------------------
    def px(self, value: float) -> int:
        return int(round(value * self.u))

    def _contains(self, x: float, y: float) -> bool:
        rx, ry, rw, rh = self.rect
        return rx <= x <= rx + rw and ry <= y <= ry + rh

    def update(self, observations: List[HandObservation]) -> Optional[int]:
        """Returns 1 as soon as the start button is fully charged."""
        self._pulse += 0.08
        centers = [(float(o.center[0]), float(o.center[1])) for o in observations]
        hovered = any(self._contains(x, y) for x, y in centers)

        if hovered:
            self.progress += 1.0 / max(self.dwell_frames, 1)
            if self.progress >= 1.0:
                return 1
        else:
            self.progress = max(0.0, self.progress - 0.05)
        return None

    def handle_key(self, key: int) -> Optional[int]:
        if key in (13, 10, 32):        # Enter or space
            return 1
        return None

    def reset(self) -> None:
        self.progress = 0.0

    # ------------------------------------------------------------------
    def render(self, image: np.ndarray) -> np.ndarray:
        canvas = image.copy()
        h, w = canvas.shape[:2]
        self.u = unit(h)
        backdrop(canvas, 0.86)

        # Wordmark: letter-spaced capitals over a fine line. The
        # letter-spacing turns two words into a logo.
        top = int(h * 0.20)
        text(canvas, "LOOP FORGE", w // 2, top, TYPE_HERO * self.u, INK, "bold",
             self.px(TRACK_HERO), align="center")
        width = text_width("LOOP FORGE", TYPE_HERO * self.u, "bold",
                           self.px(TRACK_HERO))
        cv2.line(canvas, (w // 2 - width // 2, top + self.px(16)),
                 (w // 2 + width // 2, top + self.px(16)), LINE_SOFT, 1, cv2.LINE_AA)
        label(canvas, "gesture-controlled instrument", w // 2, top + self.px(38),
              TYPE_LABEL * self.u, INK_FAINT, align="center")

        self._draw_cta(canvas)

        hint_y = int((self.rect[1] + self.rect[3]) * h) + self.px(52)
        label(canvas, "Hold your hand over the button  ·  or Enter", w // 2, hint_y,
              TYPE_MICRO * self.u, INK_SOFT, align="center")

        self._draw_gestures(canvas)
        return canvas

    def _draw_gestures(self, canvas: np.ndarray) -> None:
        """The five gestures in one line. Whoever reads this can play right away."""
        h, w = canvas.shape[:2]
        size = TYPE_MICRO * self.u
        widths = [label_width(key, size) + self.px(10)
                  + text_width(value, size, "regular")
                  for key, value in GESTURES]
        gap = self.px(30)
        total = sum(widths) + gap * (len(GESTURES) - 1)
        cursor = (w - total) // 2
        baseline = int(h * 0.87)
        cv2.line(canvas, (cursor - self.px(34), baseline - self.px(28)),
                 (cursor + total + self.px(34), baseline - self.px(28)),
                 LINE_SOFT, 1, cv2.LINE_AA)
        for (key, value), width in zip(GESTURES, widths):
            after = label(canvas, key, cursor, baseline, size, INK_SOFT)
            text(canvas, value, after + self.px(10), baseline, size,
                 scale(INK_FAINT, 0.95), "regular")
            cursor += width + gap

    def _draw_cta(self, canvas: np.ndarray) -> None:
        h, w = canvas.shape[:2]
        rx, ry, rw, rh = self.rect
        x, y = int(rx * w), int(ry * h)
        width, height = int(rw * w), int(rh * h)
        cx, cy = x + width // 2, y + height // 2
        radius = height // 2

        breathe = 0.55 + 0.45 * (0.5 + 0.5 * math.sin(self._pulse))
        active = self.progress > 0.01
        tone = ACCENT if active else mix(ACCENT_DIM, ACCENT, breathe)

        # The button breathes: a soft glow underneath that comes and goes
        # with the pulse. That draws the eye without blinking.
        self.paint.glow(canvas, cx, cy, int(width * 0.62), ACCENT,
                        0.10 + 0.16 * breathe + 0.25 * self.progress)
        self.paint.shadow(canvas, (x, y, width, height), radius,
                          spread=self.px(26), strength=0.6, drop=self.px(8))
        self.paint.glass(canvas, (x, y, width, height), radius,
                         tint=mix(PANEL, ACCENT_DIM, 0.25 + 0.3 * self.progress),
                         alpha=0.95)
        self.paint.outline(canvas, (x, y, width, height), radius, tone, 2)

        text(canvas, "START THE PARTY", cx, cy + self.px(9), TYPE_TITLE * self.u * 1.5,
             INK, "bold", self.px(4), align="center")

        # Charging as a ring on the left side of the button plus a line along
        # the bottom edge - one shows the time, the other the progress.
        if active:
            span = width - 2 * radius
            fill = int(span * clamp01(self.progress))
            cv2.line(canvas, (x + radius, y + height - 1),
                     (x + radius + fill, y + height - 1), ACCENT,
                     max(2, self.px(3)), cv2.LINE_AA)
            ring = int(radius * 0.42)
            cv2.circle(canvas, (x + radius, cy), ring, scale(INK, 0.18), 1, cv2.LINE_AA)
            arc(canvas, x + radius, cy, ring, self.progress, ACCENT, max(2, self.px(3)))


def countdown_overlay(image: np.ndarray, seconds_left: float) -> np.ndarray:
    """Short countdown so that people can line up before the start."""
    canvas = image.copy()
    h, w = canvas.shape[:2]
    u = unit(h)
    backdrop(canvas, 0.58)

    seconds = max(1, int(math.ceil(seconds_left)))
    # The ring goes round once within the current second.
    fraction = 1.0 - (seconds_left - math.floor(seconds_left))
    radius = int(min(w, h) * 0.11)
    cx, cy = w // 2, int(h * 0.47)

    paint = Painter()
    paint.glow(canvas, cx, cy, int(radius * 1.9), ACCENT, 0.22)
    cv2.circle(canvas, (cx, cy), radius, scale(INK, 0.16), 1, cv2.LINE_AA)
    arc(canvas, cx, cy, radius, fraction, ACCENT, max(2, int(3 * u)))

    text(canvas, str(seconds), cx, cy + int(radius * 0.42), TYPE_HERO * u * 1.9,
         INK, "bold", align="center")
    label(canvas, "about to start", cx, cy + radius + int(48 * u),
          TYPE_LABEL * u, INK_SOFT, align="center", tracking=int(3 * u))
    return canvas
