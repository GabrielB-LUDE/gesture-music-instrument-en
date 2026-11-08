#!/usr/bin/env python3
"""Shows what the instrument makes of the `loops/` folder.

Before a performance it is worth a look: here you can see which file was
recognised as which beat, how many bars it spans, how many
density levels it has and whether it had to be stretched for the target tempo.

    python tools/check_loops.py
    python tools/check_loops.py --bpm 140
    python tools/check_loops.py --dir /path/to/loops
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio import loops  # noqa: E402
from src.config import Config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Check your own loops")
    parser.add_argument("--bpm", type=float, help="Target tempo (otherwise from config.json)")
    parser.add_argument("--dir", default=loops.LOOPS_DIR, help="Folder with the loops")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    config = Config.load(args.config)
    bpm = args.bpm or config.audio.bpm
    samplerate = config.audio.samplerate

    library = loops.load_library(samplerate, bpm, args.dir)
    print(f"Folder   {os.path.abspath(args.dir)}")
    print(f"Tempo    {bpm:.1f} BPM, {samplerate} Hz\n")

    for problem in library.problems:
        print(f"  ! skipped - {problem}")
    if library.problems:
        print()

    if not library.presets:
        print("No loops found. The instrument uses the built-in beats.")
        print("Instructions: loops/README.md")
        return 0

    header = f"{'BEAT':<16}{'SHORT':<9}{'BARS':>6}{'LEVELS':>8}{'LENGTH':>9}  FAMILY"
    print(header)
    print("-" * len(header))
    for item in library.presets:
        clips = library.bank.get(item.bank, [])
        seconds = len(clips[0]) / samplerate if clips else 0.0
        print(
            f"{item.label[:15]:<16}{item.short:<9}{item.bars:>6}"
            f"{item.variants:>8}{seconds:>8.2f}s  {item.family}"
        )

    print(f"\n{len(library.presets)} beats in the library.")
    missing = [item.label for item in library.presets if item.variants < 4]
    if missing:
        print(
            "Fewer than four files (the finger count can only change little "
            "there): " + ", ".join(missing)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
