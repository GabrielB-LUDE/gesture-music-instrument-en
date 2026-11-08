"""Display for stage and projector.

The interface is kept calm: dark frosted glass, hairlines, lots of
space, a single accent colour. Colour only carries meaning, never
decoration - every preset has exactly one accent, and it appears as a
narrow bar, not as a colourful circle.

    left      the library: a column of slots, each one a beat,
              directly grabbable - no menu to browse through
    right     the rack: the same column again, this is where the
              self-built loops sit and can be taken out again
    at every  a card: held loop with its controls
    hand
    bottom    a line that reminds you of the gestures

Both columns have fixed, large slots. Before, they were a single
row of small circles that got tighter with every placed loop - so
hitting one was a matter of luck.

Everything visible - colours, type, glass, shadows - comes from
`theme.py`. The start screen uses the same building blocks, so that both
screens feel like one device and not like two programs.

The sizes are governed by `self.u`, the scaling factor of the
display (1.0 at 720p). Every pixel value goes through `px()`, which is why
the interface looks the same in a window as in fullscreen.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..config import Config
from ..layout import Column
from ..mapping import tokens as tk
from ..mapping.presets import colour_of, label_of, preset, short_of
from ..vision.types import HAND_CONNECTIONS, BrowserEntry, EngineSnapshot, HandView
from .theme import (
    ACCENT,
    DANGER,
    INK,
    INK_FAINT,
    INK_SOFT,
    LINE,
    LINE_SOFT,
    PANEL,
    PANEL_HI,
    RADIUS,
    TRACK_TITLE,
    TYPE_HEAD,
    TYPE_LABEL,
    TYPE_MICRO,
    TYPE_TITLE,
    Painter,
    arc,
    clamp01,
    ease_out,
    label,
    label_width,
    meter,
    mix,
    scale,
    text,
    text_width,
    unit,
)

STEPS = 16

# The gesture line at the bottom edge. Keep it short: it should remind, not
# explain.
LEGEND = (
    ("Pinch", "take / place"),
    ("Hold pinch", "delete"),
    ("Tilt", "louder"),
    ("Finger", "density"),
    ("Thumb", "tone"),
    ("Turn", "hold"),
    ("Together", "fuse"),
    ("Open / fist", "riser / drop"),
)


class Display:
    def __init__(self, config: Config):
        self.config = config
        self.window = config.ui.window_name
        self.fullscreen = config.ui.fullscreen
        self.paint = Painter()
        self._created = False
        self._vignette: Optional[np.ndarray] = None
        self._anchors: Dict[int, Tuple[int, int, int]] = {}
        self.u = 1.0

    # ------------------------------------------------------------------
    def px(self, value: float) -> int:
        """Convert a pixel size to the current image size."""
        return int(round(value * self.u))

    # ------------------------------------------------------------------
    def _ensure_window(self) -> None:
        if self._created:
            return
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        if self.fullscreen:
            cv2.setWindowProperty(
                self.window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
        self._created = True

    def toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        self._ensure_window()
        cv2.setWindowProperty(
            self.window,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL,
        )

    # ------------------------------------------------------------------
    def render(
        self,
        image: np.ndarray,
        snapshot: EngineSnapshot,
        status: Optional[str] = None,
    ) -> np.ndarray:
        canvas = image.copy()
        h, w = canvas.shape[:2]
        self.u = unit(h)
        step = self._current_step()
        pulse = self._beat_pulse()

        self._damp(canvas)
        for hand in snapshot.hands:
            self._draw_skeleton(canvas, hand)

        self._draw_library(canvas, snapshot)
        self._draw_rack(canvas, snapshot, step, pulse)
        self._draw_trash(canvas, snapshot)
        self._draw_fusion_link(canvas, snapshot)

        by_token = {token.id: token for token in snapshot.tokens}
        for hand, token, rect, card_unit in self._layout(snapshot, by_token, w, h):
            self._draw_loop_card(canvas, hand, token, rect, card_unit, step)

        for token in snapshot.tokens:
            if token.state == tk.FUSING:
                self._draw_spiral(canvas, token)

        self._draw_header(canvas, snapshot, status)
        self._draw_legend(canvas)
        return canvas

    # ------------------------------------------------------------------
    def _current_step(self) -> int:
        beats = time.time() * self.config.audio.bpm / 60.0
        return int(beats * 4.0) % STEPS

    def _beat_pulse(self) -> float:
        beats = time.time() * self.config.audio.bpm / 60.0
        return math.exp(-5.0 * (beats % 1.0))

    def _damp(self, canvas: np.ndarray) -> None:
        """Push the camera image back so that the interface sits in front.

        The vignette fades towards the edges. That keeps the
        attention in the centre, where the hands are working, and the
        cards stand out cleanly from the background.
        """
        h, w = canvas.shape[:2]
        if self._vignette is None or self._vignette.shape[:2] != (h, w):
            ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
            xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
            radial = np.sqrt(xs * xs + ys * ys) / 1.414
            fade = (0.50 - 0.32 * radial).astype(np.float32)
            self._vignette = cv2.merge([fade, fade, fade])
        cv2.multiply(canvas, self._vignette, dst=canvas, dtype=cv2.CV_8U)

    def _draw_skeleton(self, canvas: np.ndarray, hand: HandView) -> None:
        if not self.config.ui.draw_landmarks or hand.landmarks is None:
            return
        h, w = canvas.shape[:2]
        points = [(int(p[0] * w), int(p[1] * h)) for p in hand.landmarks]
        shade = scale(INK, 0.30) if hand.holding else scale(INK, 0.18)
        for a, b in HAND_CONNECTIONS:
            cv2.line(canvas, points[a], points[b], shade, 1, cv2.LINE_AA)
        for index in (4, 8, 12, 16, 20):
            cv2.circle(canvas, points[index], self.px(2), shade, -1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # The two columns
    # ------------------------------------------------------------------
    def _library_column(self, count: int) -> Column:
        cfg = self.config.forge
        return Column.packed(cfg.library_x, cfg.library_top, cfg.library_bottom,
                             count)

    def _rack_column(self) -> Column:
        cfg = self.config.forge
        return Column.packed(cfg.rack_x, cfg.rack_top, cfg.rack_bottom,
                             cfg.max_parked)

    def _slot_rect(self, column: Column, index: int, width: float,
                   w: int, h: int) -> Tuple[int, int, int, int]:
        """The rectangle of a slot. The same formula as in the forge,
        which is why the hand hits exactly what it sees."""
        # The slot spans the full slot height - that is also the size of
        # the area in which the hand hits it. It is drawn
        # capped: with only two or three beats a tile would otherwise be
        # half the screen tall and look like a bug.
        pitch = column.pitch * h
        tile_h = max(min(int(pitch - self.px(10)), self.px(92)), self.px(34))
        tile_w = int(w * width)
        cx = int(column.x * w)
        cy = int(column.slot_y(index) * h)
        return (cx - tile_w // 2, cy - tile_h // 2, tile_w, tile_h)

    def _column_title(self, canvas: np.ndarray, column: Column, width: float,
                      w: int, h: int, name: str, note: str) -> None:
        rect = self._slot_rect(column, 0, width, w, h)
        x, y, tile_w, _ = rect
        baseline = y - self.px(13)
        label(canvas, name, x + self.px(2), baseline, TYPE_LABEL * self.u,
              INK_FAINT, tracking=self.px(3))
        label(canvas, note, x + tile_w - self.px(2), baseline,
              TYPE_MICRO * self.u, scale(INK_FAINT, 0.85), align="right")

    def _hovered(self, snapshot: EngineSnapshot, zone: str) -> Dict[int, HandView]:
        """Which hand points at which slot of this column."""
        out: Dict[int, HandView] = {}
        for hand in snapshot.hands:
            if not hand.holding and hand.zone == zone:
                out.setdefault(hand.selection, hand)
        return out

    def _pointer(self, canvas: np.ndarray, hand: HandView, rect, w: int, h: int,
                 colour: Sequence[int]) -> None:
        """Hairline from the hand to the selected slot."""
        x, y, tile_w, tile_h = rect
        # From the grip point, not from the centre of the hand: that way the line
        # points exactly from the fingers to the slot they hit.
        hx, hy = int(hand.aim_x * w), int(hand.aim_y * h)
        edge = (x + tile_w, y + tile_h // 2) if hx > x + tile_w else (x, y + tile_h // 2)
        cv2.line(canvas, (hx, hy), edge, scale(colour, 0.35), 1, cv2.LINE_AA)
        cv2.circle(canvas, edge, self.px(3), scale(colour, 0.8), -1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Library on the left
    # ------------------------------------------------------------------
    def _draw_library(self, canvas: np.ndarray, snapshot: EngineSnapshot) -> None:
        """The library is a fixed column on the left edge of the image, one
        slot per beat - not a list to browse through.

        An empty hand close enough to the column points at the
        slot at its height; a pinch there takes the beat along. As with
        the trash bin, it is deliberately the position of the hand that
        decides here, not its pose.
        """
        entries = snapshot.library
        if not entries:
            return
        h, w = canvas.shape[:2]
        width = self.config.forge.library_width
        column = self._library_column(len(entries))
        self._column_title(canvas, column, width, w, h, "Library",
                           f"{len(entries)} Beats")

        hovered = self._hovered(snapshot, tk.LIBRARY)
        for index, entry in enumerate(entries):
            rect = self._slot_rect(column, index, width, w, h)
            self._draw_preset_tile(canvas, entry, rect, hovered.get(index))
        for index, hand in hovered.items():
            if index < len(entries):
                self._pointer(canvas, hand, self._slot_rect(column, index, width, w, h),
                              w, h, colour_of(entries[index].preset_ids[0]))

    def _draw_preset_tile(self, canvas: np.ndarray, entry: BrowserEntry,
                          rect: Tuple[int, int, int, int],
                          hand: Optional[HandView]) -> None:
        x, y, tile_w, tile_h = rect
        colour = colour_of(entry.preset_ids[0])
        found = preset(entry.preset_ids[0])
        active = hand is not None
        progress = hand.take_progress if hand is not None else 0.0
        radius = self.px(RADIUS - 3)

        if active:
            self.paint.glow(canvas, x + tile_w // 2, y + tile_h // 2,
                            int(tile_w * 0.7), colour, 0.10 + 0.28 * progress)
        self.paint.card(
            canvas, rect, radius,
            border=mix(LINE, colour, 0.75) if active else LINE,
            tint=PANEL_HI if active else PANEL,
            spread=self.px(12), strength=0.40,
        )
        cv2.rectangle(canvas, (x + self.px(9), y + self.px(10)),
                      (x + self.px(12), y + tile_h - self.px(10)),
                      scale(colour, 1.0 if active else 0.7), -1)

        pad = self.px(23)
        middle = y + tile_h // 2
        # With many own loops the column gets tight. Then the
        # short name stands alone - better one readable line than two
        # that touch each other.
        roomy = tile_h >= self.px(46)
        text(canvas, short_of(entry.preset_ids[0]), x + pad,
             middle - self.px(2) if roomy else middle + self.px(6),
             TYPE_HEAD * self.u, INK if active else INK_SOFT, "bold", TRACK_TITLE)
        if roomy:
            label(canvas, found.label, x + pad, middle + self.px(14),
                  TYPE_MICRO * self.u, INK_SOFT if active else INK_FAINT)
            self._kind_badge(canvas, found, x + tile_w - self.px(14),
                             middle - self.px(6), colour, active)
        elif found.kind == "loop":
            label(canvas, f"{found.variants}×", x + tile_w - self.px(14),
                  middle + self.px(6), TYPE_MICRO * self.u, INK_FAINT, align="right")
        if progress > 0.01:
            meter(canvas, x + pad, y + tile_h - self.px(9), tile_w - pad - self.px(16),
                  progress, ACCENT, self.px(2))

    def _kind_badge(self, canvas: np.ndarray, found, right: int, y: int,
                    colour: Sequence[int], active: bool) -> None:
        """On the right of the slot: for own loops, bar count and density levels.

        This is the only place where you can see how many variants
        a loop you brought yourself has - i.e. how much the finger count
        can change it at all.
        """
        if found.kind != "loop":
            return
        label(canvas, f"{found.bars} bar" + ("e" if found.bars > 1 else ""),
              right, y, TYPE_MICRO * self.u, INK_FAINT, align="right")
        pip, gap = self.px(7), self.px(3)
        total = 4 * pip + 3 * gap
        left = right - total
        for index in range(4):
            lit = index < found.variants
            cv2.rectangle(
                canvas, (left + index * (pip + gap), y + self.px(8)),
                (left + index * (pip + gap) + pip, y + self.px(10)),
                scale(colour, 0.9 if active else 0.6) if lit else LINE, -1,
            )

    # ------------------------------------------------------------------
    # Rack on the right
    # ------------------------------------------------------------------
    def _draw_rack(
        self, canvas: np.ndarray, snapshot: EngineSnapshot, step: int, pulse: float
    ) -> None:
        """The rack: fixed slots for the self-built loops.

        A placed loop stays in its slot and keeps playing
        there. An empty hand at its height brings it back with a
        short pinch; holding the pinch deletes it.
        """
        h, w = canvas.shape[:2]
        cfg = self.config.forge
        column = self._rack_column()
        width = cfg.rack_width
        slots = snapshot.rack
        used = sum(1 for entry in slots if entry is not None)
        self._column_title(canvas, column, width, w, h, "Rack",
                           f"{used} / {len(slots)}")

        by_token = {token.id: token for token in snapshot.tokens}
        hovered = self._hovered(snapshot, tk.RACK)
        for index, entry in enumerate(slots):
            rect = self._slot_rect(column, index, width, w, h)
            hand = hovered.get(index)
            if entry is None:
                self._draw_empty_slot(canvas, rect, hand)
                continue
            self._draw_rack_tile(canvas, entry, by_token.get(entry.token_id), rect,
                                 hand, pulse)
        for index, hand in hovered.items():
            if index < len(slots) and slots[index] is not None:
                self._pointer(canvas, hand, self._slot_rect(column, index, width, w, h),
                              w, h, colour_of(slots[index].preset_ids[0]))

        for token in snapshot.tokens:
            if token.state == tk.FLYING:
                self._draw_flying(canvas, token, column, width)

    def _draw_empty_slot(self, canvas: np.ndarray, rect, hand: Optional[HandView]) -> None:
        x, y, tile_w, tile_h = rect
        self.paint.outline(canvas, rect, self.px(RADIUS - 3),
                           scale(LINE, 1.4 if hand is not None else 0.8))
        label(canvas, "empty", x + tile_w // 2, y + tile_h // 2 + self.px(4),
              TYPE_MICRO * self.u, scale(INK_FAINT, 0.7), align="center")

    def _draw_rack_tile(self, canvas: np.ndarray, entry: BrowserEntry, token,
                        rect, hand: Optional[HandView], pulse: float) -> None:
        x, y, tile_w, tile_h = rect
        colour = colour_of(entry.preset_ids[0])
        active = hand is not None
        deleting = hand.delete_progress if hand is not None else 0.0
        loudness = 0.0 if token is None else max(
            (r.volume for r in token.recipes), default=0.0
        )
        border = mix(LINE, colour, 0.75) if active else LINE
        if deleting > 0.01:
            border = mix(border, DANGER, deleting)
        radius = self.px(RADIUS - 3)

        self.paint.glow(canvas, x + tile_w // 2, y + tile_h // 2, int(tile_w * 0.7),
                        DANGER if deleting > 0.01 else colour,
                        max(0.05 + 0.10 * pulse * loudness,
                            0.12 + 0.4 * deleting if active else 0.0))
        self.paint.card(canvas, rect, radius, border=border,
                        tint=PANEL_HI if active else PANEL,
                        spread=self.px(12), strength=0.40)
        cv2.rectangle(canvas, (x + self.px(9), y + self.px(10)),
                      (x + self.px(12), y + tile_h - self.px(10)),
                      scale(colour, 0.8 + 0.4 * pulse * loudness), -1)

        pad = self.px(23)
        middle = y + tile_h // 2
        name = " ".join(short_of(pid) for pid in entry.preset_ids)
        text(canvas, name, x + pad, middle - self.px(2), TYPE_HEAD * self.u,
             INK if active else INK_SOFT, "bold", TRACK_TITLE)
        if len(entry.preset_ids) > 1:
            label(canvas, f"{len(entry.preset_ids)} Beats", x + pad, middle + self.px(14),
                  TYPE_MICRO * self.u, ACCENT)
        else:
            label(canvas, label_of(entry.preset_ids[0]), x + pad, middle + self.px(14),
                  TYPE_MICRO * self.u, INK_SOFT if active else INK_FAINT)

        bar = tile_w - pad - self.px(16)
        if deleting > 0.01:
            label(canvas, "delete", x + tile_w - self.px(14), middle - self.px(6),
                  TYPE_MICRO * self.u, DANGER, align="right")
            meter(canvas, x + pad, y + tile_h - self.px(9), bar, deleting, DANGER,
                  self.px(2))
        elif active:
            label(canvas, "take", x + tile_w - self.px(14), middle - self.px(6),
                  TYPE_MICRO * self.u, ACCENT, align="right")
            meter(canvas, x + pad, y + tile_h - self.px(9), bar,
                  hand.take_progress, ACCENT, self.px(2))
        else:
            meter(canvas, x + pad, y + tile_h - self.px(9), bar, loudness,
                  scale(colour, 0.8), self.px(2))

    def _draw_trash(self, canvas: np.ndarray, snapshot: EngineSnapshot) -> None:
        """Trash bin below the rack: carrying a held loop here
        and holding deletes it."""
        h, w = canvas.shape[:2]
        cfg = self.config.forge
        cx, cy = int(cfg.trash_x * w), int(cfg.trash_y * h)
        radius = self.px(26)
        progress = max((t.trash_progress for t in snapshot.tokens if t.held), default=0.0)
        tone = mix(INK_SOFT, DANGER, progress)

        if progress > 0.01:
            self.paint.glow(canvas, cx, cy, radius * 3, DANGER, 0.45 * progress)
        cv2.circle(canvas, (cx, cy), radius, scale(INK, 0.22), 1, cv2.LINE_AA)
        if progress > 0.01:
            arc(canvas, cx, cy, radius, progress, DANGER, self.px(3))

        # Simple bin icon: lid plus container.
        bin_w, bin_h = self.px(15), self.px(17)
        top = cy - bin_h // 2 + self.px(3)
        cv2.rectangle(canvas, (cx - bin_w // 2, top), (cx + bin_w // 2, top + bin_h),
                      tone, max(1, self.px(2)), cv2.LINE_AA)
        cv2.line(canvas, (cx - bin_w // 2 - self.px(3), top),
                 (cx + bin_w // 2 + self.px(3), top), tone, max(1, self.px(2)), cv2.LINE_AA)

        label(canvas, "Trash", cx, cy + radius + self.px(17), TYPE_MICRO * self.u,
              INK_FAINT, align="center")

    def _draw_flying(self, canvas: np.ndarray, token, column: Column,
                     width: float) -> None:
        h, w = canvas.shape[:2]
        cx, cy = int(token.x * w), int(token.y * h)
        colour = colour_of(token.preset_ids[0])
        grow = ease_out(token.fly_progress)
        target = self._slot_rect(column, token.rack_index, width, w, h)
        tile_w = int(target[2] * (0.35 + 0.65 * grow))
        tile_h = int(target[3] * (0.45 + 0.55 * grow))
        rect = (cx - tile_w // 2, cy - tile_h // 2, tile_w, tile_h)
        self.paint.glow(canvas, cx, cy, int(target[2] * 0.8), colour,
                        0.35 * (1.0 - grow) + 0.1)
        self.paint.glass(canvas, rect, self.px(10), alpha=0.85)
        self.paint.outline(canvas, rect, self.px(10), mix(LINE, colour, 0.6))

    # ------------------------------------------------------------------
    # Loop in the hand
    # ------------------------------------------------------------------
    def _card_size(self, w: int, h: int, factor: float):
        """Size of the card of a held loop. `factor` shrinks it
        when there are many hands."""
        span = self._free_span(w)
        width = min(int(w * 0.21 * factor), (span[1] - span[0]) // 2)
        return width, int(h * 0.27 * factor)

    def _free_span(self, w: int) -> Tuple[int, int]:
        """The area between the two columns, in pixels."""
        cfg = self.config.forge
        left = (cfg.library_x + cfg.library_width * 0.5) * w + self.px(16)
        right = (cfg.rack_x - cfg.rack_width * 0.5) * w - self.px(16)
        return int(left), int(max(right, left + self.px(120)))

    def _layout(self, snapshot: EngineSnapshot, by_token, w: int, h: int):
        """Plan one card for every held loop, without two of them
        overlapping.

        The card sits on the outer side, away from the centre of the image: two hands
        often work close together, and cards pointing inwards would
        overlap each other.

        Then the cards on one side are pushed below each other.
        If six of them no longer fit in the image, the whole
        side shrinks evenly - better smaller cards than cards that
        cover each other. The factor goes into the card as its own unit,
        which is why the text shrinks along with it and the card stays
        consistent.

        Horizontally the card stays between the two columns. Library
        and rack are fixed parts of the interface; a card that slides
        over them would hide exactly what the second
        hand is reaching for.
        """
        gap = self.px(14)
        top_limit = int(h * 0.25)
        bottom_limit = h - self.px(64)
        available = max(bottom_limit - top_limit, 1)
        left_edge, right_edge = self._free_span(w)

        sides: Dict[bool, List] = {True: [], False: []}
        for hand in snapshot.hands:
            if not hand.holding:
                continue
            token = by_token.get(hand.token_id)
            if token is None:
                continue
            sides[hand.x < 0.5].append((hand, token))

        planned = []
        for left, members in sides.items():
            if not members:
                continue
            needed = sum(self._card_size(w, h, 1.0)[1] for _hand, _token in members)
            needed += gap * (len(members) - 1)
            factor = min(1.0, available / needed)

            group = []
            for hand, token in members:
                card_w, card_h = self._card_size(w, h, factor)
                cx, cy = int(hand.x * w), int(hand.y * h)
                offset = int(w * 0.045)
                x = cx - offset - card_w if left else cx + offset
                x = max(left_edge, min(right_edge - card_w, x))
                y = min(max(cy - card_h // 2, top_limit), bottom_limit - card_h)
                group.append([hand, token, x, y, card_w, card_h, factor])

            # Fill from the top, then pull up from the bottom. Both
            # together keep the order and let nothing stick out.
            group.sort(key=lambda item: item[3])
            cursor = top_limit
            for item in group:
                item[3] = max(item[3], cursor)
                cursor = item[3] + item[5] + gap
            cursor = bottom_limit
            for item in reversed(group):
                item[3] = min(item[3], cursor - item[5])
                cursor = item[3] - gap
            planned.extend(group)

        out = []
        for hand, token, x, y, card_w, card_h, factor in planned:
            # Light smoothing, so that the card stays still with a shaky hand.
            previous = self._anchors.get(hand.hand_id)
            if previous is not None and abs(previous[2] - card_w) < 2:
                x = int(previous[0] * 0.72 + x * 0.28)
                y = int(previous[1] * 0.72 + y * 0.28)
            self._anchors[hand.hand_id] = (x, y, card_w)
            out.append((hand, token, (x, y, card_w, card_h), self.u * factor))
        return out

    @staticmethod
    def _scaler(u: float):
        """Pixel size for a single card."""
        return lambda value: int(round(value * u))

    def _leader(self, canvas: np.ndarray, cx: int, cy: int, x: int, y: int,
                card_w: int, card_h: int, colour: Sequence[int]) -> None:
        """Thin guide line from the hand to the card."""
        anchor = (x if x > cx else x + card_w, y + card_h // 2)
        cv2.line(canvas, (cx, cy), anchor, scale(colour, 0.30), 1, cv2.LINE_AA)
        cv2.circle(canvas, anchor, self.px(3), scale(colour, 0.75), -1, cv2.LINE_AA)

    def _draw_loop_card(self, canvas: np.ndarray, hand: HandView, token,
                        rect: Tuple[int, int, int, int], u: float, step: int) -> None:
        h, w = canvas.shape[:2]
        p = self._scaler(u)
        x, y, card_w, card_h = rect
        cx, cy = int(hand.x * w), int(hand.y * h)
        colour = colour_of(token.preset_ids[0])
        frozen = token.state in (tk.FROZEN, tk.FUSING)
        recipe = token.recipes[0]
        pad = p(20)
        inner = card_w - 2 * pad

        self._leader(canvas, cx, cy, x, y, card_w, card_h, colour)
        cv2.circle(canvas, (cx, cy), self.px(22), scale(colour, 0.4), 1, cv2.LINE_AA)

        self.paint.glow(canvas, cx, cy, self.px(70), colour,
                        0.06 + 0.18 * (0.0 if token.muted else recipe.volume))
        border = mix(LINE, colour, 0.35) if frozen else LINE
        self.paint.card(canvas, rect, p(RADIUS), border=border, spread=p(20),
                        strength=0.55)

        # Header: accent, name, state.
        cv2.rectangle(canvas, (x + p(11), y + p(20)), (x + p(13), y + p(48)), colour, -1)
        name = " ".join(short_of(pid) for pid in token.preset_ids)
        text(canvas, name, x + pad, y + p(34), TYPE_TITLE * u, INK, "bold", TRACK_TITLE)
        label(canvas, label_of(token.preset_ids[0]), x + pad, y + p(50),
              TYPE_MICRO * u, INK_FAINT)
        self._state_pill(canvas, x + card_w - pad, y + p(26), token, u)

        cv2.line(canvas, (x + pad, y + p(64)), (x + card_w - pad, y + p(64)),
                 LINE_SOFT, 1, cv2.LINE_AA)

        # Controls.
        volume = 0.0 if token.muted else recipe.volume
        self._reading(canvas, x + pad, y + p(86), inner, "Volume", volume, colour, u)
        self._reading(canvas, x + pad, y + p(118), inner, "Tone", recipe.cutoff,
                      scale(colour, 0.75), u)

        found = preset(recipe.preset_id)
        label(canvas, "Density", x + pad, y + p(150), TYPE_MICRO * u, INK_FAINT)
        self._finger_pips(canvas, x + card_w - pad, y + p(145), token.fingers, colour,
                          u, usable=4 if found.kind != "loop" else found.variants)

        if found.kind == "loop":
            self._loop_strip(canvas, x + pad, y + card_h - p(36), inner, found,
                             colour, u)
        else:
            self._step_strip(canvas, x + pad, y + card_h - p(36), inner,
                             found.pattern_for(recipe.density), step, colour, u)

        # Fine line under the card: during a pinch it shows how
        # close the loop is to being thrown away, otherwise the progress of the turn.
        if token.pinch_frames > 0:
            meter(canvas, x + pad, y + card_h - p(15), inner,
                  clamp01(token.place_progress), DANGER, p(2))
        elif 0.01 < token.freeze_progress < 0.99:
            meter(canvas, x + pad, y + card_h - p(15), inner,
                  token.freeze_progress, INK_SOFT, p(2))

    def _reading(self, canvas: np.ndarray, x: int, y: int, w: int, name: str,
                 value: float, colour: Sequence[int], u: float) -> None:
        """Label, number and bar - a control as a device shows it."""
        p = self._scaler(u)
        label(canvas, name, x, y, TYPE_MICRO * u, INK_FAINT)
        text(canvas, f"{int(round(clamp01(value) * 100))}", x + w, y,
             TYPE_LABEL * u, INK_SOFT, "medium", align="right")
        meter(canvas, x, y + p(10), w, value, colour, p(3))

    def _state_pill(self, canvas: np.ndarray, right: int, y: int, token,
                    u: float) -> None:
        p = self._scaler(u)
        frozen = token.state in (tk.FROZEN, tk.FUSING)
        if token.park_blocked > 0:
            # The placing pinch went nowhere because no slot is free.
            name, tone = "Rack full", DANGER
        elif token.pinch_frames > 0:
            # The pinch is running: first the card announces placing,
            # from halfway on, throwing away.
            if token.place_progress > 0.5:
                name, tone = "discard", DANGER
            else:
                name, tone = "place", ACCENT
        elif token.muted:
            name, tone = "muted", INK_FAINT
        elif frozen:
            name, tone = "held", ACCENT
        else:
            name, tone = "open", INK_SOFT

        size = TYPE_MICRO * u
        width = label_width(name, size) + p(18)
        height = p(18)
        rect = (right - width, y - height // 2, width, height)
        self.paint.glass(canvas, rect, height // 2, tint=scale(tone, 0.18),
                         alpha=0.9, blur=False, edge=False)
        self.paint.outline(canvas, rect, height // 2, scale(tone, 0.5))
        label(canvas, name, right - width // 2, y + p(4), size, tone, align="center")

    def _finger_pips(self, canvas: np.ndarray, right: int, y: int, count: int,
                     colour: Sequence[int], u: float, usable: int = 4) -> None:
        """Four strokes for the four density levels.

        `usable` says how many of them this beat really provides: an
        own loop with only two files can only do two densities. The
        remaining strokes stay, but dark - otherwise you would look
        for the mistake in your own hand.
        """
        p = self._scaler(u)
        pip_w, gap = p(15), p(5)
        total = 4 * pip_w + 3 * gap
        x = right - total
        usable = max(1, min(4, usable))
        for index in range(4):
            left = x + index * (pip_w + gap)
            if index >= usable:
                shade = LINE_SOFT
            elif index < count:
                shade = colour
            else:
                shade = LINE
            cv2.rectangle(canvas, (left, y), (left + pip_w, y + p(3)), shade, -1)

    def _loop_strip(self, canvas: np.ndarray, x: int, y: int, w: int, found,
                    colour: Sequence[int], u: float) -> None:
        """An own loop has no individual hits but plays in one
        piece. Instead of the 16 boxes the card therefore shows its
        length with a pointer that sweeps through once per loop."""
        p = self._scaler(u)
        height = max(3, p(8))
        bars = max(found.bars, 1)
        cv2.rectangle(canvas, (x, y + height - max(1, p(2))), (x + w, y + height),
                      LINE_SOFT, -1)
        for bar in range(1, bars):
            tick = x + int(w * bar / bars)
            cv2.rectangle(canvas, (tick, y), (tick + max(1, p(1)), y + height),
                          LINE, -1)
        phase = self._loop_phase(bars)
        head = x + int(w * phase)
        cv2.rectangle(canvas, (x, y + height - max(1, p(2))), (head, y + height),
                      scale(colour, 0.8), -1)
        cv2.rectangle(canvas, (head, y), (head + max(2, p(2)), y + height), INK, -1)

    def _loop_phase(self, bars: int) -> float:
        beats = time.time() * self.config.audio.bpm / 60.0
        return (beats / (4.0 * max(bars, 1))) % 1.0

    def _step_strip(self, canvas: np.ndarray, x: int, y: int, w: int,
                    pattern: Sequence[int], step: int, colour: Sequence[int],
                    u: float) -> None:
        """The 16 sixteenths of the loop. Filled means: a hit falls here."""
        p = self._scaler(u)
        height = max(3, p(8))
        gap = max(2, p(3))
        tick = max(2, (w - gap * (STEPS - 1)) // STEPS)
        active = set(pattern)
        for index in range(STEPS):
            left = x + index * (tick + gap)
            hit = index in active
            now = index == step
            if hit:
                shade = colour if not now else INK
                cv2.rectangle(canvas, (left, y), (left + tick, y + height), shade, -1)
            else:
                top = y + height - max(1, p(2))
                cv2.rectangle(canvas, (left, top), (left + tick, y + height),
                              LINE if now else LINE_SOFT, -1)

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------
    def _draw_fusion_link(self, canvas: np.ndarray, snapshot: EngineSnapshot) -> None:
        h, w = canvas.shape[:2]
        ready = [t for t in snapshot.tokens if t.approach > 0.01 and t.held]
        if len(ready) < 2:
            return
        a, b = ready[0], ready[1]
        progress = max(a.approach, b.approach)
        ax, ay = int(a.x * w), int(a.y * h)
        bx, by = int(b.x * w), int(b.y * h)
        cv2.line(canvas, (ax, ay), (bx, by), scale(INK, 0.2 + 0.5 * progress),
                 1, cv2.LINE_AA)
        mx, my = (ax + bx) // 2, (ay + by) // 2
        self.paint.glow(canvas, mx, my, int(w * 0.04 * (0.5 + progress)), ACCENT,
                        0.4 * progress)
        label(canvas, "fuse", mx, my - int(h * 0.055), TYPE_LABEL * self.u,
              mix(INK_FAINT, INK, progress), align="center")
        ring = self.px(20)
        cv2.circle(canvas, (mx, my), ring, scale(INK, 0.2), 1, cv2.LINE_AA)
        arc(canvas, mx, my, ring, progress, INK, self.px(2))

    def _draw_spiral(self, canvas: np.ndarray, token) -> None:
        """The two loops swirl into each other, then the fusion is there."""
        h, w = canvas.shape[:2]
        progress = ease_out(token.fuse_progress)
        cx = token.fuse_center[0] + (token.x - token.fuse_center[0]) * progress
        cy = token.fuse_center[1] + (token.y - token.fuse_center[1]) * progress
        px, py = int(cx * w), int(cy * h)

        spread = 0.0
        for start_x, start_y, preset_id in token.fuse_parts:
            colour = colour_of(preset_id)
            radius0 = math.hypot(start_x - cx, start_y - cy)
            spread = max(spread, radius0)
            angle0 = math.atan2(start_y - cy, start_x - cx)
            for trail in range(8):
                local = max(0.0, progress - trail * 0.045)
                radius = radius0 * (1.0 - local)
                angle = angle0 + local * 3.4 * math.pi
                sx = int((cx + math.cos(angle) * radius) * w)
                sy = int((cy + math.sin(angle) * radius) * h)
                size = max(int(self.px(11) - trail * self.px(1.2)), 2)
                fade = 0.92 - trail * 0.1
                self.paint.glow(canvas, sx, sy, size * 3, colour, 0.42 * fade)
                cv2.circle(canvas, (sx, sy), size, scale(colour, fade), -1, cv2.LINE_AA)

        burst = int(max(spread * w, self.px(36)) * (0.4 + 1.0 * progress))
        alpha = max(0.0, 1.0 - progress) ** 0.7
        self.paint.glow(canvas, px, py, burst, INK, 0.5 * alpha)
        cv2.circle(canvas, (px, py), burst, scale(INK, alpha), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    def _draw_header(
        self, canvas: np.ndarray, snapshot: EngineSnapshot, status: Optional[str]
    ) -> None:
        h, w = canvas.shape[:2]
        held = sum(1 for t in snapshot.tokens if t.held)
        parked = sum(1 for t in snapshot.tokens if t.state == tk.PLACED)

        baseline = self.px(40)
        cursor = text(canvas, "LOOP FORGE", self.px(28), baseline,
                      TYPE_HEAD * self.u, INK, "bold", self.px(3))
        cursor += self.px(18)
        cv2.line(canvas, (cursor, baseline - self.px(12)), (cursor, baseline),
                 LINE, 1, cv2.LINE_AA)
        cursor += self.px(18)
        cursor = self._counter(canvas, cursor, baseline, str(held), "in hand")
        self._counter(canvas, cursor + self.px(22), baseline,
                      f"{parked}/{self.config.forge.max_parked}", "on rack")

        # Riser top right: a label, a bar, nothing more.
        bar_w = self.px(130)
        right = w - self.px(28)
        label(canvas, "Riser", right - bar_w - self.px(14), baseline - self.px(4),
              TYPE_MICRO * self.u, INK_FAINT, align="right")
        meter(canvas, right - bar_w, baseline - self.px(8), bar_w,
              snapshot.macros.riser, ACCENT, self.px(3))
        if snapshot.macros.drop > 0.25:
            self.paint.glow(canvas, right - bar_w // 2, baseline - self.px(8),
                            self.px(60), DANGER, 0.5 * snapshot.macros.drop)
            label(canvas, "Drop", right - bar_w // 2, baseline - self.px(20),
                  TYPE_LABEL * self.u, DANGER, align="center")

        info = f"{snapshot.fps:.0f} fps"
        if status:
            info = f"{status}   ·   {info}"
        text(canvas, info, right, baseline + self.px(20), TYPE_MICRO * self.u,
             scale(INK_FAINT, 0.9), "regular", align="right")

    def _counter(self, canvas: np.ndarray, x: int, baseline: int, value: str,
                 name: str) -> int:
        cursor = text(canvas, value, x, baseline, TYPE_HEAD * self.u, INK_SOFT,
                      "medium")
        return label(canvas, name, cursor + self.px(8), baseline,
                     TYPE_MICRO * self.u, INK_FAINT)

    def _draw_legend(self, canvas: np.ndarray) -> None:
        h, w = canvas.shape[:2]
        size = TYPE_MICRO * self.u
        widths = [label_width(key, size) + self.px(9)
                  + text_width(value, size, "regular")
                  for key, value in LEGEND]
        gap = self.px(26)
        total = sum(widths) + gap * (len(LEGEND) - 1)
        cursor = (w - total) // 2
        baseline = h - self.px(22)
        cv2.line(canvas, (0, h - self.px(48)), (w, h - self.px(48)),
                 LINE_SOFT, 1, cv2.LINE_AA)
        for (key, value), width in zip(LEGEND, widths):
            after = label(canvas, key, cursor, baseline, size, INK_SOFT)
            text(canvas, value, after + self.px(9), baseline, size,
                 scale(INK_FAINT, 0.95), "regular")
            cursor += width + gap

    # ------------------------------------------------------------------
    def show(self, canvas: np.ndarray) -> int:
        self._ensure_window()
        cv2.imshow(self.window, canvas)
        return cv2.waitKey(1) & 0xFF

    def close(self) -> None:
        if self._created:
            cv2.destroyAllWindows()
            self._created = False


def save_preview(canvas: np.ndarray, path: str) -> str:
    cv2.imwrite(path, canvas)
    return path
