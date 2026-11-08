"""Your own loop beats from the `loops/` folder.

This is where WAV files become the library that the players grab on the
left of the image. There is exactly one place to add things: the folder.
No list in the code, no configuration - whoever drops a file in gets a
new beat on the next start.

Layout
------

    loops/
        kick.wav                 one loop, one file
        bassline/                one loop with four density levels
            1_sparse.wav         1 finger
            2.wav                2 fingers
            3.wav                3 fingers
            4_full.wav           4 fingers
        loops.json               optional, only for fine-tuning

A folder is the better way: the four files in it are the same track at
four densities, and the finger count chooses between them. If there are
fewer than four files, they are spread evenly over the four levels; with
a single file the loop stays the same for every finger count.

Tempo
-----

The files do not have to be at the instrument's tempo. The number of bars
is estimated from the length and the file is then stretched exactly onto
the grid - a loop at 140 BPM then runs cleanly with the others at 128.
If you want to be precise, enter tempo and bar count in `loops.json`:

    {
      "bassline": { "label": "DEEP BASS", "short": "BAS",
                    "bpm": 140, "bars": 2, "colour": "#7AA2FF" }
    }

All fields are optional. `bpm` says at which tempo the file was
recorded, `bars` how many bars it spans.
"""

from __future__ import annotations

import json
import os
import wave
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import PROJECT_ROOT
from ..mapping.presets import LOOP_COLOURS, Preset
from .synth import load_wav_mono, normalise

LOOPS_DIR = os.path.join(PROJECT_ROOT, "loops")
META_FILE = "loops.json"

# Without a value, the bar count is estimated from the length and rounded
# to one of these values. Anything in between would almost always be a measurement error.
BAR_CHOICES = (1, 2, 4, 8)
LOOP_PEAK = 0.88
FADE_SECONDS = 0.004   # short fade at both ends against clicks

# These words in the name give a loop its family. The
# family only controls the sort order in the library.
FAMILY_HINTS = (
    ("drums", ("kick", "drum", "beat", "clap", "snare", "hat", "perc", "top")),
    ("bass", ("bass", "sub", "808", "acid")),
    ("lead", ("lead", "melo", "arp", "pluck", "synth")),
    ("chord", ("chord", "stab", "pad", "key")),
    ("fx", ("fx", "sweep", "riser", "noise", "atmo", "vox")),
)
FAMILY_ORDER = {"drums": 0, "bass": 1, "chord": 2, "lead": 3, "fx": 4, "other": 5}


@dataclass(frozen=True)
class LoopLibrary:
    """What the folder provides: presets for the display, buffers for the sound."""

    presets: List[Preset]
    bank: Dict[str, List[np.ndarray]]
    problems: List[str]

    def __bool__(self) -> bool:
        return bool(self.presets)


_CACHE: Dict[Tuple[str, int, float], LoopLibrary] = {}


# ----------------------------------------------------------------------
def load_library(
    samplerate: int, bpm: float, directory: Optional[str] = None
) -> LoopLibrary:
    """Reads the loop folder. The result is cached so that the display
    and the sound engine use the same buffers."""
    directory = directory or LOOPS_DIR
    key = (os.path.abspath(directory), int(samplerate), round(float(bpm), 4))
    cached = _CACHE.get(key)
    if cached is None:
        cached = _scan(directory, int(samplerate), float(bpm))
        _CACHE[key] = cached
    return cached


def clear_cache() -> None:
    _CACHE.clear()


# ----------------------------------------------------------------------
def _scan(directory: str, samplerate: int, bpm: float) -> LoopLibrary:
    presets: List[Preset] = []
    bank: Dict[str, List[np.ndarray]] = {}
    problems: List[str] = []

    if not os.path.isdir(directory):
        return LoopLibrary(presets, bank, problems)

    meta = _read_meta(directory, problems)
    sources = _collect_sources(directory)

    for order, (name, paths) in enumerate(sources):
        entry = meta.get(name, {})
        clips: List[np.ndarray] = []
        bars = 0
        for path in paths[:4]:
            try:
                clip, clip_bars = _load_clip(path, samplerate, bpm, entry)
            except (FileNotFoundError, wave.Error, ValueError, OSError) as exc:
                problems.append(f"{os.path.relpath(path, PROJECT_ROOT)}: {exc}")
                continue
            clips.append(clip)
            bars = max(bars, clip_bars)
        if not clips:
            continue
        # All variants of a loop must be exactly the same length, otherwise
        # the density levels drift apart when switching.
        clips = [_fit_length(clip, len(clips[0])) for clip in clips]
        bank[name] = clips
        presets.append(_make_preset(name, entry, order, bars, len(clips)))

    presets.sort(key=lambda item: (FAMILY_ORDER.get(item.family, 9), item.label))
    return LoopLibrary(presets, bank, problems)


def _read_meta(directory: str, problems: List[str]) -> Dict[str, dict]:
    path = os.path.join(directory, META_FILE)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        problems.append(f"{META_FILE}: {exc}")
        return {}
    return {str(key): value for key, value in data.items() if isinstance(value, dict)}


def _collect_sources(directory: str) -> List[Tuple[str, List[str]]]:
    """Every WAV file directly in the folder is a loop, every subfolder
    a loop with several density levels."""
    out: List[Tuple[str, List[str]]] = []
    for name in sorted(os.listdir(directory)):
        full = os.path.join(directory, name)
        if os.path.isdir(full):
            files = sorted(
                os.path.join(full, item)
                for item in os.listdir(full)
                if item.lower().endswith(".wav") and not item.startswith(".")
            )
            if files:
                out.append((_slug(name), files))
        elif name.lower().endswith(".wav") and not name.startswith("."):
            out.append((_slug(os.path.splitext(name)[0]), [full]))
    return out


# ----------------------------------------------------------------------
def _load_clip(
    path: str, samplerate: int, bpm: float, entry: dict
) -> Tuple[np.ndarray, int]:
    data = load_wav_mono(path, samplerate)
    if len(data) < 16:
        raise ValueError("File is too short")

    source_bpm = _positive(entry.get("bpm")) or bpm
    bars = int(entry.get("bars") or 0)
    if bars <= 0:
        bars = _guess_bars(len(data) / samplerate, source_bpm)

    target = int(round(bars * 4.0 * 60.0 / bpm * samplerate))
    clip = _resample(data, target)
    clip = _fade_edges(clip, samplerate)
    return normalise(clip, LOOP_PEAK), bars


def _positive(value: Optional[object]) -> Optional[float]:
    """A number from `loops.json`, provided it is usable."""
    try:
        number = float(value)   # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number > 0.0 else None


def _guess_bars(seconds: float, bpm: float) -> int:
    beats = seconds * bpm / 60.0
    raw = max(beats / 4.0, 1e-3)
    return min(BAR_CHOICES, key=lambda bars: abs(bars - raw))


def _resample(data: np.ndarray, target: int) -> np.ndarray:
    """Stretches the loop to exactly `target` samples.

    Linear interpolation is enough: the correction is usually small, and a
    clean edge towards the bar matters more than the last percent of
    sound fidelity.
    """
    target = max(target, 16)
    if len(data) == target:
        return data.astype(np.float32)
    source = np.linspace(0.0, 1.0, num=len(data), endpoint=False)
    wanted = np.linspace(0.0, 1.0, num=target, endpoint=False)
    return np.interp(wanted, source, data).astype(np.float32)


def _fit_length(clip: np.ndarray, length: int) -> np.ndarray:
    if len(clip) == length:
        return clip
    out = np.zeros(length, dtype=np.float32)
    take = min(len(clip), length)
    out[:take] = clip[:take]
    return out


def _fade_edges(clip: np.ndarray, samplerate: int) -> np.ndarray:
    n = min(int(FADE_SECONDS * samplerate), len(clip) // 8)
    if n < 2:
        return clip
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    clip = clip.copy()
    clip[:n] *= ramp
    clip[-n:] *= ramp[::-1]
    return clip


# ----------------------------------------------------------------------
def _make_preset(
    name: str, entry: dict, order: int, bars: int, variants: int
) -> Preset:
    label = str(entry.get("label") or name.replace("_", " ").upper())
    short = str(entry.get("short") or _short_of(name)).upper()[:4]
    family = str(entry.get("family") or _family_of(name))
    colour = _colour(entry.get("colour"), order)
    gain = float(entry.get("gain") or 1.0)
    return Preset(
        id=name,
        label=label,
        short=short,
        family=family,
        bank=name,
        note_mode="loop",
        # A loop needs no hit patterns. The four full grids
        # are only there so that everything that touches a preset - display,
        # OSC, tests - does not need to know about special cases.
        patterns=tuple(tuple(range(16)) for _ in range(4)),
        colour=colour,
        gain=max(0.05, min(gain, 2.0)),
        kind="loop",
        bars=max(bars, 1),
        variants=max(variants, 1),
    )


def _slug(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in name.strip().lower())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or "loop"


def _short_of(name: str) -> str:
    parts = [part for part in name.split("_") if part]
    if len(parts) >= 2:
        return (parts[0][:2] + parts[1][:1]).upper()
    return name[:3].upper()


def _family_of(name: str) -> str:
    for family, words in FAMILY_HINTS:
        if any(word in name for word in words):
            return family
    return "other"


def _colour(value: Optional[object], order: int) -> Tuple[int, int, int]:
    """Read the colour from `loops.json`, otherwise take it from the sequence."""
    if isinstance(value, str) and value.strip():
        text = value.strip().lstrip("#")
        if len(text) == 6:
            try:
                r, g, b = (int(text[i:i + 2], 16) for i in (0, 2, 4))
                return (b, g, r)     # the display draws in BGR
            except ValueError:
                pass
    if isinstance(value, (list, tuple)) and len(value) == 3:
        r, g, b = (max(0, min(255, int(part))) for part in value)
        return (b, g, r)
    return LOOP_COLOURS[order % len(LOOP_COLOURS)]
