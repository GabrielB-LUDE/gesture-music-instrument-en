# Loop Forge

A musical instrument that needs no controller. An ordinary webcam
films the room, detects up to six hands and turns them into a track
that builds up layer by layer in front of the audience.

What makes it special is not the hand tracking, but that it is played like
an instrument: **only the pose of the hand counts, never its location in
the image**. The hand stays relaxed in front of the body; nobody has to walk
across the image.

With an empty hand you grab a beat directly from the **library**
on the left of the image - a column of slots, one slot per beat - and take
it along with a **pinch**. The loop
then hangs on the hand: **extended fingers** set how dense the
beat hits (one finger sparse, four fingers full), the **tilt of the
hand** the volume, the **spread thumb** the timbre, and the
**fist** mutes it. Once the beat sits, you **turn the wrist over**
- back of the hand to the camera - and all values lock in. The loop keeps
running, and the pose is free again. Turning back releases it again.

That is exactly what holding is for: two held loops whose hands are
brought together **fuse** immediately into a single one that
carries both beats. A short **pinch** puts a loop on the **rack**
on the right of the image, where it keeps playing. The rack is the second column,
a mirror image of the library: an empty hand at the height of a slot
brings the loop back with a short pinch, a long pinch deletes
it. The same long pinch with a loop in the hand: it is gone.

The beats in the library are either the eight built-in ones or
**your own loops**: dropping WAV files into the `loops/` folder is enough,
nothing more is needed (section 4).

The result is an arrangement that stays visible in the image, and
an instrument in which two people can fuse their loops into one -
even two different people.

The project runs in two modes:

* **Standalone**: built-in sound engine, music plays immediately.
  No Ableton, no licence, no preparation.
* **DAW**: all values are also sent out via OSC and can be mapped in Ableton,
  Bitwig, Reaper or FL Studio.

---

## 1. What you need to install yourself

| What | Why | Where from |
|---|---|---|
| Python 3.10 to 3.13 | runtime | python.org or Homebrew |
| Webcam | input | built-in or USB |
| Headphones or PA | output | already there |

The project fetches everything else itself: the Python packages via
`requirements.txt` and the MediaPipe hand model on the first start
(about 8 MB, one time only; the machine briefly needs internet for that).

Optional, only for the DAW route:

* Ableton Live (or another DAW)
* `python-rtmidi` for the OSC to MIDI bridge
* on Windows, additionally loopMIDI for virtual MIDI ports

---

## 2. Up and running in two minutes

### macOS and Linux

```bash
chmod +x start_mac_linux.sh
./start_mac_linux.sh
```

### Windows

Double-click `start_windows.bat`.

### By hand

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python tools/check_setup.py       # checks camera, audio, packages
python main.py
```

On the first start, macOS asks for camera permission for the
Terminal. Without that permission the image stays black.

### Testing without a camera

```bash
python main.py --mock             # virtual hands, everything else real
python main.py --mock --skip-lobby     # without the start screen
python tools/render_demo.py demo.wav   # 16-second demo as WAV
```

---

## 3. How to play

### The posture

Stand so that your upper body and both forearms are in the image. The
hands stay loose in front of the upper body, roughly at chest height, the
palms towards the camera, the elbows close to the body. And that is where they
stay: everything is played through the pose, not the position.
That is more comfortable, looks calmer and is recognised much more reliably than
moving around in the image.

Two things help a lot:

* **Extend fingers clearly or curl them clearly.** The camera reads
  half measures as something in between, and then the density wobbles.
* **When turning, turn the whole wrist**, don't just tilt it. The
  back of the hand should face the camera flat, just like the
  palm did before. In between lies a dead zone in which nothing happens -
  so the turn can be slow.

### Controls per hand

| Gesture | Effect |
|---|---|
| **Pinch** (thumb and index finger touch) | empty hand at the library on the left: takes the beat at its height. Empty hand at the rack on the right: brings the loop back. Hand with a loop: places it |
| **Tilt the hand sideways** | only with a held loop: volume. Upright is the middle, outwards louder, inwards quieter |
| **Extend fingers (1 to 4)** | density of the beat: one finger sparse, four fingers full |
| **Fist** | mute, but the loop stays in the hand |
| **Spread the thumb** | timbre from dull to open |
| **Turn the wrist over** | locks all values. Turning back releases them again |
| **Bring both hands together** | two held loops fuse into one |
| **Short pinch with a loop in the hand** | puts the loop on the rack, the hand is free again |
| **Hold the pinch** | throws away: with a loop in the hand, the loop itself; at the rack, the slot the hand points at. The card or slot shows the progress |
| **Carry a held loop to the trash bin** | holding the hand with the loop briefly over the bin icon below the rack deletes it (see below) |
| **All hands open wide / clench into a fist together** | riser and drop |

Almost none of these gestures depend on **where** the hand is in the image - only
the two columns at the edges and the trash bin do so deliberately, because there
you see something concrete in space and reach for it (see below). Otherwise
the hand can stay relaxed in front of the body the whole time; it counts
fingers, tilts, turns and pinches. The only movement left is for fusing,
and even that is just the short distance until both hands meet.

The loop is always held with the values that sounded **before** the turn.
While turning, the palm briefly tilts onto its edge;
otherwise every held beat would lock in duller than it just was.

### The cycle

1. **Reach.** Move an empty hand to the library on the left of the image, at the height
   of the beat you want. The slot lights up and a hairline points from
   the hand to it.
2. **Take.** Pinch until the bar in the slot is full.
3. **Shape.** Fingers count the density, tilt sets the volume, the thumb
   the timbre. The card next to the hand shows everything.
4. **Hold.** Turn the wrist over. The card shows `HELD`.
5. **Fuse.** Fetch a second loop into the other hand the same way and
   hold it, then bring both hands together. The line between the
   hands fills up, after which one loop carries both beats.
6. **Place.** Pinch briefly and open again. The loop flies to the rack
   and keeps playing there. If you keep the pinch closed, you see the line
   under the card fill up red - then the loop is gone.
7. **Bring back.** Empty hand to the rack on the right, at the height of the slot.
   A short pinch - open, closed, open again - brings the loop back into the hand
   with all its values. If you hold the pinch instead, you see the bar
   in the slot fill up red and the loop is deleted.

A fused loop keeps the mix of its beats. When it is
unfrozen again, the tilt is its master control: upright is full, tilted
inwards it fades out.

### The two columns

On the left the **library** with the beats that are there to grab, on the right
the **rack** with the loops that have already been built. Both are columns
of large, equally tall slots in fixed places - no menu to
browse and, more importantly, nothing that shifts: a slot stays
where it is, even when the one above becomes empty. As with the trash bin,
it is deliberately the position of the hand that decides here, not its pose: you
see the slot and reach for it.

An empty hand that comes close enough to a column points at the
slot at its height. What counts is the height of the point where
thumb and index finger grab (`grip_reach`), not that of the
palm - after all, you aim with the fingers you are about to
close. The thin line in the image points from exactly there to the slot
that is hit.

So that the selection does not flicker at every slot boundary, a resting
hand is calmed and the selection is locked (`select_smoothing`,
`select_hysteresis`). Both only apply while the hand is still: if it moves
visibly faster than `select_settle` per frame, the selection follows
immediately, and the closing pinch takes whatever slot is under the
fingers at that moment anyway. In practice that means: point and
grab - there is no need to wait.

* **Library on the left.** One slot per beat, with short name, name and - for
  your own loops - bar count and number of density levels. A beat that is taken
  stays in the library, so several people can hold the same beat
  at the same time. The column never changes while
  playing.
* **Rack on the right.** Five slots (`max_parked` in `config.json`).
  A placed loop keeps playing there and shows its volume.
  A short pinch brings it back into the hand, a long pinch deletes it.
  If the rack is full, the card in the hand says `RACK FULL` instead of
  silently swallowing the pinch.

### The trash bin

Below the rack on the right sits a small bin icon. Like
taking from the columns, this deliberately works through the position of the hand
instead of its pose: carrying a held loop there and holding briefly
deletes it completely, with a progress ring on the card and
on the icon itself. Alternatively, the long pinch from
the table above still works.

### Group macros


Both work through the pose, not the height of the hands. Nobody
has to raise their arms above their head.

* **Riser**: all hands that are currently adjusting a loop open wide
  together - four fingers extended and the thumb clearly spread. The
  riser value visibly rises. Musically this fits together: the open
  hand is at the same time full density and open timbre.
* **Drop**: out of the full riser, everyone clenches their fist together.
  The fist mutes the loops, and the pulse hits exactly there.

### Keys in the window

`ESC` quit, `f` fullscreen, `m` mirroring, `h` skeleton on and off,
`r` reshuffle library, `space` place everything held,
`x` delete everything, `l` back to selection.

---

## 4. Your own beats instead of the built-in ones

There are two ways, and they solve two different tasks.

### 4a. Your own loops - the normal way

If you have finished loops (one bar of drums, a bassline, a top loop),
just put them into the `loops/` folder. Nothing else: no entry in the
code, no configuration. At the next start they appear on the left in the
library and replace the built-in beats.

```
loops/
    toploop.wav              -> one beat, one file
    bassline/                -> one beat with four density levels
        1_sparse.wav             1 finger
        2.wav                    2 fingers
        3.wav                    3 fingers
        4_full.wav               4 fingers
```

The folder is the better way: the files in it are the same track at
four densities, and the finger count chooses between them - exactly as
the finger count chooses between four hit patterns for the built-in beats.
With fewer than four files the levels are spread evenly;
with a single one the finger count changes nothing, and the
card in the hand shows that (only as many strokes under `DENSITY` as
there are files).

**Tempo.** The files do not have to be at the instrument's tempo.
The bar count is estimated from the length and the file is stretched exactly onto the
grid, so that everything runs together. It is more precise with a
`loops.json` next to the files:

```json
{
  "bassline": {
    "label": "DEEP BASS", "short": "BAS",
    "bpm": 140, "bars": 2,
    "colour": "#7AA2FF", "family": "bass", "gain": 0.9
  }
}
```

All fields are optional. `bpm` is the tempo of the recording, `bars`
its length in bars.

**Check beforehand.** This command shows what each file was recognised
as - bar count, density levels, length - and what was skipped:

```bash
python tools/check_loops.py
python tools/check_loops.py --bpm 140      # for a different target tempo
```

**Try it once.** This command creates two example loops, so that
you see the layout in action once before your own files
go in:

```bash
python tools/make_example_loops.py          # create
python tools/make_example_loops.py --clean  # remove again
```

**Format.** WAV, PCM 8/16/32 bit, mono or stereo, any sample rate.
MP3 and AIFF do not work; convert them to WAV first.

**Back to the built-in beats.** `python main.py --library builtin`,
or permanently `"library": "builtin"` in the `audio` section of
`config.json`. `"auto"` (default) uses your own loops as soon as
there are any, otherwise the built-in ones.

Where this lives in the code: `src/audio/loops.py` reads the folder,
`src/app.py` (`install_library`) sets the library with it, and
`src/audio/engine.py` plays a loop layer in one piece instead of from
individual hits. The controls stay the same - tilt for volume,
thumb for timbre, fingers for density.

### 4b. Replacing individual sounds of the built-in beats

The built-in beats are made up of individual hits on a
16th-note grid; the sounds for them are computed from waveforms
(`src/audio/synth.py`). If you only want to replace one of them -
say, your own kick drum while keeping the built-in patterns:

1. Put a WAV file in `samples/`, e.g. `samples/kick/my_kick.wav`.
2. Register in `src/audio/sample_files.py` under which "bank" it
   should play:

   ```python
   CUSTOM_SAMPLES = {
       "kick": ["samples/kick/my_kick.wav"],
   }
   ```

3. Restart `python main.py`. The file then automatically replaces the
   built-in sound of that bank; volume, filter and pattern of a
   preset stay unchanged, because they belong to the bank and not
   to the individual file.

Which bank names exist is listed as the `bank` field of each preset in
`src/mapping/presets.py`. The complete instructions including examples
are right at the top of `src/audio/sample_files.py`.

---

## 5. Command line

```bash
python main.py --library builtin      # ignore your own loops in loops/
python main.py --performers 2         # preset the selection
python main.py --skip-lobby           # start directly without the start screen
python main.py --countdown 5          # longer countdown
python main.py --seed 42              # fixed random seed
python main.py --camera 1             # second camera
python main.py --bpm 140              # tempo of the internal engine
python main.py --no-audio             # OSC only, internal engine off
python main.py --no-osc               # internal engine only
python main.py --fullscreen           # straight onto the projector
python main.py --mock --no-window --frames 100    # test run
python main.py --preview image.png    # save the last frame
```

Permanent settings live in `config.json` and are read at start-up.
The command line wins over the file.

---

## 6. OSC interface

The target is `127.0.0.1:9000`, all values are floats from 0 to 1. There is
one address group per preset from the crate:

```
/stem/kick_four/volume     volume
/stem/kick_four/cutoff     filter
/stem/kick_four/density    pattern density
/stem/kick_four/active     1 = this loop is currently in play
/global/riser              build-up, 0 to 1
/global/drop               short pulse
/global/hands              number of detected hands
/global/tokens             number of loops in play
/global/placed             of those, placed
/global/heartbeat          always running, for checking the connection
```

The same applies to all sixteen preset IDs: `kick_four`, `kick_broken`, `clap`,
`snare`, `hats_closed`, `hats_trap`, `rim`, `toms`, `bass_sub`,
`bass_acid`, `lead_pluck`, `arp`, `chord_stab`, `pad`, `sweep`, `blip`.
If the same preset runs in several tokens, the loudest one wins.

To see whether anything arrives:

```bash
python tools/osc_monitor.py 9000
```

DAW connection: see `max_for_live/README.md`.

---

## 7. Setting up at the venue

* Camera on a tripod, roughly at chest height, tilted slightly downwards.
  The crate sits in the bottom fifth of the image; you need to be able to reach
  down there comfortably with your hand.
* Three to four metres of distance, so that three people side by side with
  outstretched arms fit into the image.
* Projector image as large as possible. Mirrored is the default, so that the
  movement in the image matches your own movement.
* Light from the front. Backlight is the most common reason for poor
  tracking; a window behind the players kills the detection.
* Projector on the extended display, then `f` for fullscreen.
* On weak hardware: `--width 960 --height 540` is completely sufficient.

---

## 8. Project structure

```
gesture-music-instrument/
├── main.py                  entry point
├── config.json              default settings
├── requirements.txt
├── start_mac_linux.sh       setup and start in one
├── start_windows.bat
├── src/
│   ├── app.py               arguments, main loop, keys
│   ├── config.py            configuration
│   ├── engine.py            hands to snapshot
│   ├── layout.py            geometry of the two columns
│   ├── vision/
│   │   ├── camera.py        camera in its own thread
│   │   ├── tracker.py       MediaPipe hand landmarker and mock
│   │   ├── hand_tracker.py  stable IDs per hand across frames
│   │   └── types.py         shared data types
│   ├── mapping/
│   │   ├── presets.py       the built-in beats and the active library
│   │   ├── tokens.py        library, holding, fusion, rack
│   │   ├── gestures.py      fingers, tilt, thumb, wrist turn
│   │   ├── normalizer.py    smoothing
│   │   └── macros.py        riser and drop
│   ├── network/osc_bridge.py
│   ├── audio/
│   │   ├── engine.py        sequencer with any number of layers
│   │   ├── loops.py         your own loops from the loops/ folder
│   │   ├── sample_files.py  your own single sounds for the built-in beats
│   │   └── synth.py         sound generation and filters
│   └── ui/
│       ├── theme.py         colours, type, glass - for both screens
│       ├── display.py       projector image: library, rack, hand cards
│       └── lobby.py         start screen
├── loops/                   your own loop WAVs go in here
├── tools/
│   ├── check_loops.py       check your own loops
│   ├── make_example_loops.py  two example loops to try out
│   ├── check_setup.py       system check
│   ├── osc_monitor.py       watch OSC traffic
│   ├── osc_to_midi.py       bridge to any DAW
│   └── render_demo.py       demo WAV without a camera
├── tests/                   113 tests, run without a camera
└── max_for_live/            instructions and Max patch
```

---

## 9. Tests

```bash
python -m unittest discover -s tests -v
```

The 113 tests run completely without a camera, without a sound card and without the model.
They cover pose recognition (finger count, tilt, thumb,
wrist turn, pinch), including proof that the values are
independent of the position in the image; re-identification of individual
hands across several frames; the complete cycle of library, holding,
fusion and rack - including taking out of and deleting from the rack and
the lock against a flickering selection -; all presets; the path of your own
loops from the WAV in the folder to the sound; OSC sending to a real
local receiver; and the sound engine including its computing-time budget. On top of that
comes a run through both screens: game display, start screen and
countdown are each drawn completely into a test image once, and the
hand cards must not overlap even with six hands.

---

## 10. When something doesn't work

| Symptom | Cause and solution |
|---|---|
| `Camera 0 cannot be opened` | other index: `--camera 1`, or camera permissions in the system settings |
| Black image on macOS | Terminal needs camera permission; restart Terminal afterwards |
| Model doesn't load | download the file manually and put it at `models/hand_landmarker.task`; the URL is in `src/config.py` |
| No sound | `pip install sounddevice`; on Linux additionally `sudo apt install libportaudio2` |
| Sound crackles | `"blocksize": 512` in `config.json` |
| Pinch is not recognised | close the fingers more clearly, or `"pinch_max": 0.38` in `config.json` |
| The column selects a slot above or below the fingers | adjust `"grip_reach"` in `config.json`: larger aims higher, smaller lower (default 1.3) |
| Selection jumps back and forth while aiming | `"select_hysteresis": 0.30` or `"select_settle": 0.020` in `config.json` |
| Selection lags behind the hand | `"select_smoothing": 0.2` or `"select_settle": 0.008` in `config.json` |
| Loop doesn't freeze | turn the wrist further, or `"flip_threshold": 0.18` in `config.json` |
| Loop is accidentally placed while grabbing | `"place_frames": 6` in `config.json` |
| Short pinch doesn't place anything | `"place_frames": 3` in `config.json` |
| Loop disappears too quickly | `"discard_frames": 30` in `config.json` |
| Riser doesn't trigger | spread the thumb more clearly, or `"riser_open": 0.78` |
| Fusion doesn't trigger | bring the hands closer together, or `"fuse_distance": 0.20` |
| Frozen value doesn't fit | `"flip_arm": 0.45` in `config.json` |
| Values jitter | `"smoothing": 0.5` in `config.json` |
| Text looks jagged | Pillow is missing: `pip install pillow` (then a real system font instead of the OpenCV stroke font) |
| Low FPS | `--width 960 --height 540`, make the window smaller |
| Nothing arrives in the DAW | first `python tools/osc_monitor.py`, then check the port |
