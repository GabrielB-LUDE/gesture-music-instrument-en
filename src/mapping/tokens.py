"""The beat forge.

A loop is a beat in the hand. You play exclusively with the pose of the
hand - the hand can stay relaxed in front of the body and never has to
move anywhere:

    library (left)        empty hand at the height of a slot, pinch takes
    LIVE                  fingers count the density, tilt the
                          volume, the thumb the timbre
    FROZEN                turning the wrist over locks the values,
                          turning it back releases them again
    FUSING                bring two held loops together
    FLYING / PLACED       a short pinch puts the loop on the rack, where it
                          keeps playing
    rack (right)          empty hand at the height of a slot: a short pinch
                          brings the loop back into the hand, a long pinch
                          deletes it

The two columns have fixed slots in fixed places. Before, library and
rack shared one list; every placed loop shifted and shrank everything in
it, until hitting the right one became a game of chance. Separate columns
with a fixed slot height and a lock against switching (`Column.pick`)
make aiming reliable again.

Aiming uses `gestures.grip_point`, i.e. the point where thumb and index
finger grab - not the tracked centre of the hand, which sits roughly one
slot height below. And the lock only applies to the resting hand: as soon
as the hand visibly moves, smoothing and lock are released, and the
closing pinch snaps to the height of the fingers without either.
Otherwise the column still points at a slot the hand has long passed,
and you grab the wrong one next to it.

A loop can carry several recipes. Each recipe is a preset with its own
frozen values, which is why a fusion sounds exactly like the two single
loops before it.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

from ..config import ForgeConfig, MappingConfig
from ..layout import Column
from ..vision.hand_tracker import HandTrack
from ..vision.types import BrowserEntry, HandView, SoundSlot
from . import gestures
from .normalizer import ChannelSmoother
from .presets import PRESET_IDS, preset

LIVE = "live"
FROZEN = "frozen"
FUSING = "fusing"
FLYING = "flying"
PLACED = "placed"

# The two columns at the edges of the image
LIBRARY = "library"
RACK = "rack"


@dataclass
class Recipe:
    """A preset with its values. The key stays stable across fusions,
    so that the sound engine does not have to recreate the voice."""

    key: str
    preset_id: str
    volume: float = 0.5
    cutoff: float = 0.55
    density: float = 0.375


@dataclass
class Token:
    id: int
    recipes: List[Recipe]
    state: str = LIVE
    x: float = 0.5
    y: float = 0.5
    hand_id: Optional[int] = None
    track: Optional[HandTrack] = field(default=None, repr=False)
    muted: bool = False
    hold_frames: int = 0
    fingers: int = 2                # last adopted finger count
    freeze_progress: float = 0.0    # 0 = open, 1 = held
    flip_frames: int = 0            # how long the hand side has been right
    count_candidate: int = -1
    count_frames: int = 0
    approach: float = 0.0           # closeness to the fusion partner, 0..1
    fuse_progress: float = 0.0      # animation progress, 0..1
    fuse_parts: List[Tuple[float, float, str]] = field(default_factory=list)
    fuse_center: Tuple[float, float] = (0.5, 0.5)
    pinch_frames: int = 0           # how long the placing pinch has been held
    place_progress: float = 0.0     # 0 = open, 1 = about to be thrown away
    pinch_latched: bool = True      # the grabbing pinch must open first
    trash_hold_frames: int = 0      # how long the loop has been in the bin
    trash_progress: float = 0.0     # 0 = away from the bin, 1 = about to be deleted
    rack_index: int = 0
    park_blocked: int = 0           # remaining frames of the "rack full" message
    fly_progress: float = 0.0
    fly_from: Tuple[float, float] = (0.5, 0.5)
    # Master control: only used for fused loops.
    master: float = 1.0
    # Values from the moment when the hand was still flat to the camera.
    hold_values: Tuple[float, float, float] = (0.5, 0.55, 0.375)
    hold_master: float = 1.0
    smoother: ChannelSmoother = field(default_factory=ChannelSmoother, repr=False)

    @property
    def held(self) -> bool:
        return self.hand_id is not None

    @property
    def visible_hand(self) -> bool:
        return self.track is not None and self.track.visible

    @property
    def fused(self) -> bool:
        return len(self.recipes) > 1

    @property
    def preset_ids(self) -> List[str]:
        return [recipe.preset_id for recipe in self.recipes]


@dataclass
class BrowserHand:
    """The state of an empty hand at one of the two columns."""

    hand_id: int
    zone: str = LIBRARY
    index: int = 0
    pinch_frames: int = 0
    cooldown: int = 0
    latched: bool = False           # pinch must be opened once
    pinching: bool = False          # has the grip been closed since the last frame?
    smooth_y: float = -1.0          # calmed target height for the selection
    delete_progress: float = 0.0    # 0 = take out, 1 = about to be deleted


class TokenForge:
    def __init__(
        self,
        forge: ForgeConfig,
        mapping: MappingConfig,
        seed: Optional[int] = None,
    ):
        self.cfg = forge
        self.mapping = mapping
        self.rng = random.Random(seed)
        self.dt = 1.0 / 30.0
        self.tokens: List[Token] = []
        self.library: List[str] = list(PRESET_IDS)
        self.events: List[str] = []
        self.hands: List[HandView] = []
        self.browsers: Dict[int, BrowserHand] = {}
        self._next_token = 1
        self._next_recipe = 1
        self._fuse_hold: Dict[Tuple[int, int], int] = {}

    # ------------------------------------------------------------------
    def shuffle_library(self) -> None:
        self.rng.shuffle(self.library)

    def place_all(self) -> None:
        """All held loops move to the rack."""
        for token in list(self.tokens):
            if token.held:
                self._park(token, instant=True)

    def clear(self) -> None:
        self.tokens.clear()
        self.browsers.clear()
        self._fuse_hold.clear()
        self.hands = []

    # ------------------------------------------------------------------
    @property
    def library_column(self) -> Column:
        """The left column: one slot per beat in the library."""
        cfg = self.cfg
        return Column.packed(cfg.library_x, cfg.library_top, cfg.library_bottom,
                             len(self.library))

    @property
    def rack_column(self) -> Column:
        """The right column: always `max_parked` slots.

        The number of slots deliberately does not depend on how many
        loops are in it. A slot stays where it was, even when the one
        above becomes free - otherwise you would have to search again
        every time you reach for it.
        """
        cfg = self.cfg
        return Column.packed(cfg.rack_x, cfg.rack_top, cfg.rack_bottom,
                             cfg.max_parked)

    def library_entries(self) -> List[BrowserEntry]:
        """The beats that are available to grab fresh."""
        return [
            BrowserEntry(kind="preset", preset_ids=[preset_id])
            for preset_id in self.library
        ]

    def rack_entries(self) -> List[BrowserEntry]:
        """One entry per slot in the rack, empty slots as None."""
        slots: List[Optional[BrowserEntry]] = [None] * max(self.cfg.max_parked, 1)
        for token in self.parked:
            if 0 <= token.rack_index < len(slots):
                slots[token.rack_index] = BrowserEntry(
                    kind="loop", preset_ids=token.preset_ids, token_id=token.id
                )
        return slots

    def entries(self) -> List[BrowserEntry]:
        """Everything that can be grabbed - library first. Only for overview
        and tests; playing goes through the two separate columns."""
        return self.library_entries() + [
            entry for entry in self.rack_entries() if entry is not None
        ]

    @property
    def parked(self) -> List[Token]:
        return [t for t in self.tokens if t.state == PLACED]

    def token_by_id(self, token_id: Optional[int]) -> Optional[Token]:
        return next((t for t in self.tokens if t.id == token_id), None)

    def _free_rack_slot(self) -> int:
        """The topmost free slot. -1 if the rack is full."""
        taken = {token.rack_index for token in self.parked}
        taken.update(
            token.rack_index for token in self.tokens if token.state == FLYING
        )
        for index in range(max(self.cfg.max_parked, 1)):
            if index not in taken:
                return index
        return -1

    # ------------------------------------------------------------------
    def update(self, tracks: List[HandTrack], dt: float = 1.0 / 30.0) -> List[Token]:
        self.events.clear()
        self.dt = max(dt, 1e-3)
        by_id = {track.id: track for track in tracks}
        busy: Set[int] = set()

        # 1. attach held loops to their hand
        for token in list(self.tokens):
            if not token.held:
                continue
            track = by_id.get(token.hand_id)
            token.track = track
            if track is None or track.missing > self.cfg.release_frames:
                self._park(token, instant=True)
                if token.held:
                    # The hand is gone and there was no free slot in the rack.
                    # Otherwise the loop would have kept sounding invisibly,
                    # without anyone being able to reach it.
                    self._release_hand(token)
                    self._remove(token)
                    self.events.append("discard")
                continue
            busy.add(track.id)
            token.hold_frames += 1
            if track.visible:
                token.x = float(track.position[0])
                token.y = float(track.position[1])

        # 2. advance the state of each loop
        for token in list(self.tokens):
            token.park_blocked = max(0, token.park_blocked - 1)
            if token.state == LIVE:
                self._update_live(token)
            elif token.state == FROZEN:
                self._update_frozen(token)
            elif token.state == FUSING:
                self._update_fusing(token)
            elif token.state == FLYING:
                self._update_flying(token)

        # 3. bin: carrying a held loop into the corner deletes it
        self._update_trash()

        # 4. check fusion
        self._update_fusion()

        # 5. free hands browse the library
        free = [track for track in tracks if track.visible and track.id not in busy]
        self._update_browser(free, {track.id for track in tracks})

        # 6. realign the rack and feed the display
        self._layout_rack()
        self._collect_hands(tracks, by_id)
        return self.tokens

    # ------------------------------------------------------------------
    # Loop in the hand
    # ------------------------------------------------------------------
    def _update_live(self, token: Token) -> None:
        if not token.visible_hand:
            return
        obs = token.track.observation
        # During the placing pinch the index finger curls in. If the controls
        # kept running, the density would drop at the same moment - the loop
        # would sound different when placed than when played. That is why
        # all values rest during the pinch.
        if self._check_release(token):
            return
        count = self._stable_count(token, obs)
        token.muted = count == 0
        lean = gestures.lean(obs, self.mapping)

        if token.fused:
            # A fused loop keeps the mix of its beats. The tilt
            # becomes the master control: upright is full, tilted
            # inwards fades out. So upright means "nothing changes" -
            # which is why nothing jumps when unfreezing.
            token.master = token.smoother.smooth("master", min(1.0, lean * 2.0))
        else:
            recipe = token.recipes[0]
            recipe.volume = token.smoother.smooth("volume", lean)
            recipe.cutoff = token.smoother.smooth(
                "cutoff", gestures.thumb_open(obs, self.mapping)
            )
            recipe.density = gestures.density_from_count(count)

        # As long as the palm clearly faces the camera, the values count
        # as played. As soon as the wrist turns, this state stays
        # put - it is what locks in when holding.
        if gestures.palm_facing(obs) > self.mapping.flip_arm:
            first = token.recipes[0]
            token.hold_values = (first.volume, first.cutoff, first.density)
            token.hold_master = token.master

        # Turn the wrist over: lock the values.
        if self._flip_holds(token, obs, want_back=True):
            self._freeze(token)
            return

        token.freeze_progress = max(
            0.0, min(0.9, token.flip_frames / max(self.cfg.freeze_frames, 1))
        )

    def _update_frozen(self, token: Token) -> None:
        if not token.held or not token.visible_hand:
            return
        obs = token.track.observation
        # Showing the palm again releases the loop. In a
        # fusion the individual beats stay as they are - there only
        # the master control still applies.
        if self._flip_holds(token, obs, want_back=False):
            token.state = LIVE
            token.freeze_progress = 0.0
            token.master = 1.0
            token.hold_master = 1.0
            token.hold_values = (
                token.recipes[0].volume,
                token.recipes[0].cutoff,
                token.recipes[0].density,
            )
            token.smoother.reset_to(token.recipes[0].volume, token.recipes[0].cutoff)
            token.smoother.filters.pop("master", None)
            self.events.append("unfreeze")
            return
        token.freeze_progress = 1.0
        self._check_release(token)

    def _update_fusing(self, token: Token) -> None:
        token.fuse_progress += 1.0 / max(self.cfg.fuse_duration, 1)
        if token.fuse_progress >= 1.0:
            token.fuse_progress = 1.0
            token.state = FROZEN
            token.fuse_parts = []

    def _update_flying(self, token: Token) -> None:
        token.fly_progress += 1.0 / max(self.cfg.fly_frames, 1)
        target = self._rack_position(token.rack_index)
        ease = 1.0 - (1.0 - min(token.fly_progress, 1.0)) ** 3
        token.x = token.fly_from[0] + (target[0] - token.fly_from[0]) * ease
        token.y = token.fly_from[1] + (target[1] - token.fly_from[1]) * ease
        if token.fly_progress >= 1.0:
            token.state = PLACED
            token.x, token.y = target

    # ------------------------------------------------------------------
    def _stable_count(self, token: Token, obs) -> int:
        """Finger count with a short settling period.

        Without this lock a single frame with a half-extended
        finger would briefly switch the density - you hear that immediately.
        """
        count = gestures.finger_count(obs, self.mapping)
        if count == token.fingers:
            token.count_candidate = -1
            token.count_frames = 0
            return token.fingers
        if count != token.count_candidate:
            token.count_candidate = count
            token.count_frames = 0
        token.count_frames += 1
        if token.count_frames >= max(self.mapping.finger_hold_frames, 1):
            token.fingers = count
            token.count_candidate = -1
            token.count_frames = 0
        return token.fingers

    def _flip_holds(self, token: Token, obs, want_back: bool) -> bool:
        """Counts how long the wrist has been on the requested side."""
        facing = gestures.palm_facing(obs)
        threshold = self.mapping.flip_threshold
        matched = facing < -threshold if want_back else facing > threshold
        if matched:
            token.flip_frames += 1
        else:
            token.flip_frames = 0
        return token.flip_frames >= max(self.mapping.flip_frames, 1)

    def _freeze(self, token: Token) -> None:
        """Lock in the values - with the state from before the turn.

        While turning, the palm tilts onto its edge. The visible
        thumb distance shrinks, and with it the timbre. Without
        this fallback every held loop would sound duller than
        the one you were just playing.
        """
        if token.fused:
            for recipe in token.recipes:
                recipe.volume = min(1.0, recipe.volume * token.hold_master)
            token.master = 1.0
            token.hold_master = 1.0
        else:
            recipe = token.recipes[0]
            recipe.volume, recipe.cutoff, recipe.density = token.hold_values
        token.state = FROZEN
        token.freeze_progress = 1.0
        token.flip_frames = 0
        token.muted = False
        self.events.append("freeze")

    # ------------------------------------------------------------------
    # Placing
    # ------------------------------------------------------------------
    def _check_release(self, token: Token) -> bool:
        """Pinch with a loop in the hand: short = place, long = throw away.

        It triggers when the fingers open, not when they close.
        That way you can still see on the card where the loop is going, and
        you can cancel an accidentally started pinch by releasing it
        immediately.

        The pinch that just fetched the loop does not count:
        it is locked until the fingers open once.
        Otherwise a single long pinch would take the loop and throw
        it away again in the same move.

        Returns: True while a pinch is held - then the controls rest,
        so that the curled index finger does not change the density.
        """
        if not token.visible_hand:
            token.pinch_frames = 0
            token.place_progress = 0.0
            return False

        pinching = gestures.is_pinch(token.track.observation, self.mapping)
        if not pinching:
            held = token.pinch_frames
            latched = token.pinch_latched
            token.pinch_frames = 0
            token.place_progress = 0.0
            token.pinch_latched = False
            if not latched and held >= max(self.cfg.place_frames, 1):
                self._park(token)
                return True
            return False

        # While two loops move towards each other, the hand counts as busy
        # bringing them together. Otherwise the loop slips out of the hand
        # while fusing.
        if token.pinch_latched or token.approach > 0.01 \
                or token.hold_frames < self.cfg.min_hold_frames:
            token.pinch_frames = 0
            token.place_progress = 0.0
            return True

        token.pinch_frames += 1
        span = max(self.cfg.discard_frames - self.cfg.place_frames, 1)
        token.place_progress = min(
            1.0, max(0, token.pinch_frames - self.cfg.place_frames) / span
        )
        if token.pinch_frames >= max(self.cfg.discard_frames, 1):
            self._release_hand(token)
            self._remove(token)
            self.events.append("discard")
        return True

    def _update_trash(self) -> None:
        """Bin next to the rack: carrying a loop there and holding deletes it.

        Everything else in the instrument only reads the pose of the hand,
        never its location in the image - that is deliberate. This one
        action is the exception: deleting should feel like carrying
        something away, not like having to remember yet another finger pose.
        """
        cfg = self.cfg
        for token in list(self.tokens):
            if not token.held or token.state not in (LIVE, FROZEN):
                token.trash_hold_frames = 0
                token.trash_progress = 0.0
                continue
            inside = math.hypot(token.x - cfg.trash_x, token.y - cfg.trash_y) <= cfg.trash_radius
            if inside:
                token.trash_hold_frames += 1
            else:
                token.trash_hold_frames = 0
            token.trash_progress = min(
                1.0, token.trash_hold_frames / max(cfg.trash_frames, 1)
            )
            if token.trash_hold_frames >= max(cfg.trash_frames, 1):
                self._release_hand(token)
                self._remove(token)
                self.events.append("discard")

    def _park(self, token: Token, instant: bool = False) -> None:
        """The loop moves to the rack and keeps playing there."""
        slot = self._free_rack_slot()
        if slot < 0:
            # No free slot. The loop stays in the hand, and the card
            # says why for a moment - silently doing nothing felt
            # like a bug in the instrument.
            token.pinch_frames = 0
            token.place_progress = 0.0
            token.park_blocked = 45
            self.events.append("rack_full")
            return
        if token.state == LIVE:
            self._freeze(token)
        self._release_hand(token)
        token.rack_index = slot
        token.fly_from = (token.x, token.y)
        token.fly_progress = 0.0
        token.pinch_frames = 0
        token.place_progress = 0.0
        token.pinch_latched = True
        token.state = FLYING
        if instant:
            token.fly_progress = 1.0
            token.state = PLACED
            token.x, token.y = self._rack_position(token.rack_index)
        self.events.append("park")

    def _release_hand(self, token: Token) -> None:
        """Detach the loop from the hand and briefly lock the hand.

        Without the lock the hand that just became free would land in the
        library at the same moment and immediately grab something again
        on the next pinch.
        """
        if token.hand_id is not None:
            browser = self.browsers.get(token.hand_id)
            if browser is None:
                browser = BrowserHand(hand_id=token.hand_id)
                self.browsers[token.hand_id] = browser
            browser.cooldown = self.cfg.take_cooldown
            browser.latched = True
            browser.pinch_frames = 0
        token.hand_id = None
        token.track = None
        token.hold_frames = 0

    def _remove(self, token: Token) -> None:
        if token in self.tokens:
            self.tokens.remove(token)

    def _layout_rack(self) -> None:
        for token in self.parked:
            token.x, token.y = self._rack_position(token.rack_index)

    def _rack_position(self, index: int) -> Tuple[float, float]:
        column = self.rack_column
        index = max(0, min(index, column.slots - 1))
        return column.x, column.slot_y(index)

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------
    def _update_fusion(self) -> None:
        ready = [
            token
            for token in self.tokens
            if token.state == FROZEN and token.held and token.visible_hand
        ]
        for token in self.tokens:
            token.approach = 0.0

        pairs = sorted(
            (
                (math.hypot(a.x - b.x, a.y - b.y), a, b)
                for a, b in combinations(ready, 2)
            ),
            key=lambda item: item[0],
        )

        seen: Set[Tuple[int, int]] = set()
        busy: Set[int] = set()
        for distance, a, b in pairs:
            if a.id in busy or b.id in busy:
                continue
            if distance > self.cfg.fuse_distance:
                continue
            if len(a.recipes) + len(b.recipes) > self.cfg.max_recipes:
                continue
            busy.add(a.id)
            busy.add(b.id)
            key = (min(a.id, b.id), max(a.id, b.id))
            seen.add(key)
            held = self._fuse_hold.get(key, 0) + 1
            self._fuse_hold[key] = held
            # Closeness and hold duration together make up the visible indicator.
            closeness = 1.0 - distance / max(self.cfg.fuse_distance, 1e-6)
            progress = min(1.0, held / max(self.cfg.fuse_frames, 1))
            a.approach = b.approach = max(progress, closeness * 0.6)
            if held >= self.cfg.fuse_frames:
                self._begin_fusion(a, b)

        for key in list(self._fuse_hold):
            if key not in seen:
                del self._fuse_hold[key]

    def _begin_fusion(self, a: Token, b: Token) -> None:
        keeper, donor = (a, b) if a.id <= b.id else (b, a)
        keeper.fuse_parts = [
            (keeper.x, keeper.y, keeper.recipes[0].preset_id),
            (donor.x, donor.y, donor.recipes[0].preset_id),
        ]
        keeper.fuse_center = ((keeper.x + donor.x) * 0.5, (keeper.y + donor.y) * 0.5)
        keeper.recipes = keeper.recipes + donor.recipes
        keeper.state = FUSING
        keeper.fuse_progress = 0.0
        keeper.approach = 0.0
        self._release_hand(donor)
        self._remove(donor)
        self._fuse_hold.pop((min(a.id, b.id), max(a.id, b.id)), None)
        self.events.append("fuse")

    # ------------------------------------------------------------------
    # The two columns
    # ------------------------------------------------------------------
    def _update_browser(self, free: List[HandTrack], alive: Set[int]) -> None:
        cfg = self.cfg
        # The state belongs to the hand, not to whether it is currently free.
        # Otherwise the lock after placing would be lost again in the same
        # frame.
        for hand_id in list(self.browsers):
            if hand_id not in alive:
                del self.browsers[hand_id]

        library = self.library_entries()
        rack = self.rack_entries()

        for track in free:
            obs = track.observation
            if obs is None:
                self.browsers.pop(track.id, None)
                continue

            # Aiming uses the point where thumb and index finger
            # grab, not the tracked centre of the hand. That sits
            # roughly one slot height lower - you pointed at one slot and
            # got the one below.
            aim = gestures.grip_point(obs, self.mapping)
            x, y = float(aim[0]), float(aim[1])
            if x <= cfg.library_capture_x and library:
                zone, column = LIBRARY, self.library_column
            elif x >= cfg.rack_capture_x and any(rack):
                zone, column = RACK, self.rack_column
            else:
                self.browsers.pop(track.id, None)
                continue

            browser = self.browsers.get(track.id)
            if browser is None:
                browser = BrowserHand(hand_id=track.id, zone=zone)
                self.browsers[track.id] = browser
            if browser.zone != zone:
                # Changing sides: reset everything, so that no half
                # pinch carries over from the other column.
                browser.zone = zone
                browser.smooth_y = -1.0
                browser.pinch_frames = 0
                browser.pinching = False
                browser.delete_progress = 0.0
                browser.latched = True
            browser.cooldown = max(0, browser.cooldown - 1)

            # The target height points at a slot - the same formula as for
            # drawing. Smoothing and lock keep a resting hand on
            # its slot; they should not slow down a hand that is visibly
            # moving. That is why both are released with
            # speed: `step` is the distance since the last frame,
            # `settled` drops from 1 (hand still) to 0 (hand sweeping
            # through).
            if browser.smooth_y < 0.0:
                browser.smooth_y = y
                browser.index = column.nearest(y)
            step = abs(y - browser.smooth_y)
            settled = max(0.0, 1.0 - step / max(cfg.select_settle, 1e-6))
            keep = min(max(cfg.select_smoothing, 0.0), 0.95) * settled
            browser.smooth_y += (y - browser.smooth_y) * (1.0 - keep)

            pinching = gestures.is_pinch(obs, self.mapping)
            # A grip that actually triggers something. At the moment
            # it closes, only what is under the fingers counts: no
            # lock, no lag. After that the selection is fixed
            # as long as the grip holds - otherwise the slot would still drift
            # away while you are already grabbing it.
            armed = pinching and not browser.latched and browser.cooldown <= 0
            if armed and not browser.pinching:
                browser.smooth_y = y
                browser.index = column.nearest(y)
            elif not armed:
                browser.index = column.pick(
                    browser.smooth_y, browser.index,
                    cfg.select_hysteresis * settled,
                )
            browser.pinching = armed

            if zone == LIBRARY:
                self._browse_library(browser, library, track, pinching)
            else:
                self._browse_rack(browser, rack, track, pinching)

    def _browse_library(
        self, browser: BrowserHand, entries: List[BrowserEntry],
        hand: HandTrack, pinching: bool,
    ) -> None:
        """Left: a pinch takes the beat along immediately."""
        if not pinching:
            browser.latched = False
            browser.pinch_frames = 0
            return
        if browser.latched or browser.cooldown > 0:
            return
        browser.pinch_frames += 1
        if browser.pinch_frames >= max(self.cfg.take_frames, 1):
            self._take(entries[min(browser.index, len(entries) - 1)], hand, browser)

    def _browse_rack(
        self, browser: BrowserHand, slots: List[Optional[BrowserEntry]],
        hand: HandTrack, pinching: bool,
    ) -> None:
        """Right: a short pinch brings the loop back, a long one deletes it.

        The same rule as with a loop in the hand - short moves,
        long throws away. It triggers when the fingers open, so
        a started grip can be cancelled by releasing it
        immediately.
        """
        cfg = self.cfg
        entry = slots[browser.index] if browser.index < len(slots) else None
        if not pinching:
            held, latched = browser.pinch_frames, browser.latched
            browser.latched = False
            browser.pinch_frames = 0
            browser.delete_progress = 0.0
            if (
                not latched
                and browser.cooldown <= 0
                and entry is not None
                and held >= max(cfg.take_frames, 1)
            ):
                self._take(entry, hand, browser)
            return

        if browser.latched or browser.cooldown > 0 or entry is None:
            browser.pinch_frames = 0
            browser.delete_progress = 0.0
            return

        browser.pinch_frames += 1
        span = max(cfg.discard_frames - cfg.take_frames, 1)
        browser.delete_progress = min(
            1.0, max(0, browser.pinch_frames - cfg.take_frames) / span
        )
        if browser.pinch_frames >= max(cfg.discard_frames, 1):
            self._delete_parked(entry, browser)

    def _delete_parked(self, entry: BrowserEntry, browser: BrowserHand) -> None:
        browser.pinch_frames = 0
        browser.delete_progress = 0.0
        browser.latched = True
        browser.cooldown = self.cfg.take_cooldown
        token = self.token_by_id(entry.token_id)
        if token is None or token.state != PLACED:
            return
        self._remove(token)
        self.events.append("discard")

    def _take(self, entry: BrowserEntry, hand: HandTrack, browser: BrowserHand) -> None:
        browser.pinch_frames = 0
        browser.delete_progress = 0.0
        browser.latched = True
        browser.cooldown = self.cfg.take_cooldown
        if entry.kind == "loop":
            token = self.token_by_id(entry.token_id)
            if token is None or token.state != PLACED:
                return
            token.state = FROZEN
            token.hand_id = hand.id
            token.track = hand
            token.hold_frames = 0
            token.freeze_progress = 1.0
            token.flip_frames = 0
            token.pinch_frames = 0
            token.place_progress = 0.0
            token.pinch_latched = True
            token.x = float(hand.position[0])
            token.y = float(hand.position[1])
            self.events.append("pickup")
            return
        if len(self.tokens) >= self.cfg.max_tokens:
            self.events.append("full")
            return
        self._spawn(entry.preset_ids[0], hand)

    def _spawn(self, preset_id: str, hand: HandTrack) -> Token:
        recipe = Recipe(key=f"r{self._next_recipe}", preset_id=preset_id)
        self._next_recipe += 1
        token = Token(
            id=self._next_token,
            recipes=[recipe],
            state=LIVE,
            x=float(hand.position[0]),
            y=float(hand.position[1]),
            hand_id=hand.id,
            track=hand,
        )
        token.smoother.alpha = self.mapping.smoothing
        if hand.observation is not None:
            token.fingers = gestures.finger_count(hand.observation, self.mapping)
        self._next_token += 1
        self.tokens.append(token)
        self.events.append("grab")
        return token

    # ------------------------------------------------------------------
    def _collect_hands(self, tracks: List[HandTrack], by_id: Dict[int, HandTrack]) -> None:
        held = {t.hand_id: t for t in self.tokens if t.held}
        views: List[HandView] = []
        for track in tracks:
            if not track.visible or track.observation is None:
                continue
            obs = track.observation
            token = held.get(track.id)
            browser = self.browsers.get(track.id)
            aim = gestures.grip_point(obs, self.mapping)
            views.append(
                HandView(
                    hand_id=track.id,
                    x=float(track.position[0]),
                    y=float(track.position[1]),
                    aim_x=float(aim[0]),
                    aim_y=float(aim[1]),
                    handedness=track.handedness,
                    holding=token is not None,
                    token_id=token.id if token else None,
                    zone=browser.zone if browser else "",
                    selection=browser.index if browser else 0,
                    take_progress=(
                        min(1.0, browser.pinch_frames / max(self.cfg.take_frames, 1))
                        if browser
                        else 0.0
                    ),
                    delete_progress=browser.delete_progress if browser else 0.0,
                    fingers=token.fingers if token else gestures.finger_count(obs, self.mapping),
                    facing=gestures.palm_facing(obs),
                    lean=gestures.lean(obs, self.mapping),
                    thumb=gestures.thumb_open(obs, self.mapping),
                    landmarks=obs.landmarks,
                )
            )
        self.hands = views

    # ------------------------------------------------------------------
    def slots(self) -> List[SoundSlot]:
        out: List[SoundSlot] = []
        for token in self.tokens:
            gain = 0.0 if token.muted else token.master
            for recipe in token.recipes:
                out.append(
                    SoundSlot(
                        key=recipe.key,
                        preset_id=recipe.preset_id,
                        gain=recipe.volume * gain * preset(recipe.preset_id).gain,
                        cutoff=recipe.cutoff,
                        density=recipe.density,
                    )
                )
        return out

    @property
    def live_openness(self) -> List[float]:
        """Hand openness of the players who are currently shaping a loop.

        The group macros build their build-up from this: all hands
        opening wide together is a riser, all clenching into a fist together
        is the drop. The position in the image no longer plays any
        role anywhere.
        """
        return [
            gestures.openness(token.track.observation, self.mapping)
            for token in self.tokens
            if token.state == LIVE and not token.muted and token.visible_hand
        ]

    @property
    def placed_count(self) -> int:
        return len(self.parked)
