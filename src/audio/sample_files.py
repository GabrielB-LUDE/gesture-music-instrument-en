"""Your own sounds instead of the built-in synthesis.

**How to add your own sound:**

1. Put a WAV file (PCM, any sample rate) into the `samples/` folder,
   ideally in one subfolder per sound, e.g. `samples/kick/my_kick.wav`.
2. Register it below in `CUSTOM_SAMPLES`: under which bank it should
   play (see the list of bank names further down) and with which path,
   relative to the project folder.
3. On the next start (`python main.py`) the file is loaded and
   automatically replaces the built-in buffers of that bank - volume,
   filter and pattern of a preset stay unchanged, because they belong
   to the bank, not to the individual audio file.

Several paths in a list are allowed, e.g. several variants for
"hats" or several notes for "sub" - the same selection logic as for the
built-in sounds then applies here too (see `pick_sample` in
`src/audio/engine.py`).

**Which bank names exist** (defined in `src/mapping/presets.py`,
field `bank` of each preset): kick, clap, hats, sub, acid, lead, stab,
sweep. Creating a new preset with a new bank name works the same way -
just add a new entry with its own `bank` name to `PRESETS` in
`presets.py` and register the matching file here.

Example:

    CUSTOM_SAMPLES = {
        "kick": ["samples/kick/my_kick.wav"],
        "clap": ["samples/clap/clap_a.wav", "samples/clap/clap_b.wav"],
    }
"""

from __future__ import annotations

from typing import Dict, List

CUSTOM_SAMPLES: Dict[str, List[str]] = {
    # "kick": ["samples/kick/my_kick.wav"],
}
