# Connecting to Ableton Live

There are three ways to control the DAW. The first needs nothing at all,
the third sounds the most professional.

## Way 1: internal sound engine (no DAW needed)

Default. `python main.py` starts and plays immediately. Good for the hackathon,
because nothing can go wrong and no machine additionally needs Ableton.

## Way 2: OSC to MIDI (any DAW)

```bash
pip install python-rtmidi
python tools/osc_to_midi.py          # Terminal 1
python main.py --no-audio            # Terminal 2
```

The script opens a virtual MIDI output called `GestureInstrument`.
In Ableton, under Preferences, Link/MIDI, set the input to Remote.
Then press Cmd+M (Windows: Ctrl+M), click a control and
move a hand. Mapped.

MIDI channels:

| Channel | Instrument |
|---|---|
| 1 | Kick |
| 2 | Bass |
| 3 | Snare |
| 4 | Hi-Hats |
| 5 | Synth Lead |
| 6 | Synth Pad |

CC per channel: 1 = volume, 74 = filter, 2 = pattern density, 7 = mute.
CC 11 on channel 1 is the riser.

Windows has no virtual MIDI ports out of the box. For that,
install [loopMIDI](https://www.tobias-erichsen.de/software/loopmidi.html)
first and create a port there.

## Way 3: build the Max for Live device yourself

An `.amxd` file is a binary format and cannot sensibly be shipped as
text. But the device is built in two minutes:

1. In Ableton, create a MIDI track, drag `Max Audio Effect` onto it, and click
   the pencil. Max opens.
2. Delete everything in the patch, then copy the contents of `GestureOSCReceiver.txt`
   and paste them into Max with Cmd+V. Max builds the finished
   patch from it.
3. Save as `GestureOSCReceiver.amxd`.
4. Drag the device onto each of the six tracks. At the top of the device, set the slot and the
   hand, then use the Map button to map it to volume and filter.

The patch listens on UDP port 9000 and distributes the addresses
`/stem/<instrument>/...` to Live parameters. The instruments are called
`kick`, `bass`, `snare`, `hats`, `synth1` and `synth2`.
