#!/usr/bin/env python3
"""Writes two example loops into the `loops/` folder.

Meant for trying things out: afterwards run `python main.py` once, and
the library shows the two files from the folder instead of the built-in
beats. That way you see in one minute how your own loops are integrated,
before you drop in your own files.

    python tools/make_example_loops.py          # create
    python tools/make_example_loops.py --clean  # remove again

The generated files are deliberately simple (a kick pattern at four
densities and a bassline over two bars) - they demonstrate the layout,
not good taste.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio.loops import LOOPS_DIR  # noqa: E402

SR = 44100
BPM = 128.0
BAR = 4.0 * 60.0 / BPM
NAMES = ("drumloop", "bassline_deep.wav")


def write_wav(path: str, data: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    clipped = np.clip(data, -1.0, 1.0)
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes((clipped * 32000).astype("<i2").tobytes())


def drum_bar(level: int) -> np.ndarray:
    """One bar of kick drum; the higher `level`, the more hits."""
    n = int(BAR * SR)
    out = np.zeros(n)
    hits = ((0, 8), (0, 4, 8, 12), (0, 4, 8, 12, 14), tuple(range(0, 16, 2)))[level]
    step = BAR / 16.0
    for position in hits:
        start = int(position * step * SR)
        length = min(int(0.20 * SR), n - start)
        t = np.arange(length) / SR
        pitch = 48.0 + 95.0 * np.exp(-32.0 * t)
        out[start:start + length] += np.sin(2 * np.pi * pitch * t) * np.exp(-13.0 * t)
    return out * 0.8


def bass_two_bars() -> np.ndarray:
    """Two bars of bassline, eight notes."""
    notes = (55.0, 55.0, 73.4, 65.4, 55.0, 82.4, 73.4, 65.4)
    n = int(BAR * 2 * SR)
    out = np.zeros(n)
    seg = n // len(notes)
    for index, freq in enumerate(notes):
        t = np.arange(seg) / SR
        tone = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(4 * np.pi * freq * t)
        out[index * seg:index * seg + seg] = tone * np.exp(-2.2 * t)
    return out * 0.5


def main() -> int:
    parser = argparse.ArgumentParser(description="Create example loops")
    parser.add_argument("--clean", action="store_true", help="Delete the examples again")
    parser.add_argument("--dir", default=LOOPS_DIR)
    args = parser.parse_args()

    if args.clean:
        for name in NAMES:
            path = os.path.join(args.dir, name)
            if os.path.isdir(path):
                shutil.rmtree(path)
            elif os.path.exists(path):
                os.remove(path)
        print(f"Examples removed from {args.dir}.")
        return 0

    for level in range(4):
        write_wav(os.path.join(args.dir, "drumloop", f"{level + 1}.wav"),
                  drum_bar(level))
    write_wav(os.path.join(args.dir, "bassline_deep.wav"), bass_two_bars())
    print(f"Examples created in {args.dir}:")
    print("  drumloop/1..4.wav    one bar, four density levels")
    print("  bassline_deep.wav    two bars, one level")
    print("\nCheck:  python tools/check_loops.py")
    print("Start:  python main.py")
    print("Remove: python tools/make_example_loops.py --clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
