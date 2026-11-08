# Your own loop beats

Whatever is in here shows up in the library on the left at the next start and
replaces the built-in beats. Nothing more is needed - no line of code,
no entry in a list.

## One file, one beat

    loops/kick.wav          ->  beat "KICK"
    loops/toploop.wav       ->  beat "TOPLOOP"

## One folder, one beat with four density levels (recommended)

    loops/bassline/
        1_sparse.wav        ->  1 finger
        2.wav               ->  2 fingers
        3.wav               ->  3 fingers
        4_full.wav          ->  4 fingers

The files are sorted by name; the first one belongs to the lowest
density. If there are fewer than four, they are spread evenly over the
four finger counts; with a single file the finger count changes
nothing - the card in the hand then shows that too.

## Tempo

The files do not have to be at the instrument's tempo. The bar count is
estimated from the length and the file is stretched exactly onto the grid.
If you want to specify it precisely, put a `loops.json` next to them:

    {
      "bassline": {
        "label": "DEEP BASS",
        "short": "BAS",
        "bpm": 140,
        "bars": 2,
        "colour": "#7AA2FF",
        "family": "bass",
        "gain": 0.9
      }
    }

All fields are optional.

## Check before you start

    python tools/check_loops.py

shows for every file what it was recognised as - bar count, variants,
tempo correction. Use `--bpm 140` for a different target tempo.

## Format

WAV, PCM 8/16/32 bit, mono or stereo, any sample rate. MP3 and AIFF
do not work - convert them to WAV first.

## Back to the built-in beats

    python main.py --library builtin

Or `"library": "builtin"` in the `audio` section of `config.json`.
