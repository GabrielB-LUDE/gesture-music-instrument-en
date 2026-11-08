#!/usr/bin/env python3
"""Bridge from OSC to MIDI CC.

This lets you control any DAW, even without Max for Live: the script
opens a virtual MIDI output and sends the hand values as
control change messages. In the DAW, just use MIDI Learn.

Additionally requires:  pip install python-rtmidi

Usage:  python tools/osc_to_midi.py [--port 9000] [--channel 1]

CC assignment (MIDI channels 1 to 6 = the six instruments):
    CC 1  volume
    CC 74 filter cutoff
    CC 2  pattern density
    CC 7  mute (0 or 127)
    CC 11 riser (channel 1 only)
"""

from __future__ import annotations

import argparse
import threading
import time

CHANNEL_MAP = {
    "/stem/kick": 0,
    "/stem/bass": 1,
    "/stem/snare": 2,
    "/stem/hats": 3,
    "/stem/synth1": 4,
    "/stem/synth2": 5,
}
CC_MAP = {"volume": 1, "cutoff": 74, "density": 2, "mute": 7}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--name", default="GestureInstrument")
    args = parser.parse_args()

    try:
        import rtmidi
    except ImportError:
        print("python-rtmidi is missing.  pip install python-rtmidi")
        return 1

    from pythonosc.dispatcher import Dispatcher
    from pythonosc.osc_server import ThreadingOSCUDPServer

    midi_out = rtmidi.MidiOut()
    try:
        midi_out.open_virtual_port(args.name)
        print(f"Virtual MIDI output open: {args.name}")
    except Exception:
        ports = midi_out.get_ports()
        if not ports:
            print("No MIDI output available. On Windows, install loopMIDI first.")
            return 2
        midi_out.open_port(0)
        print(f"MIDI output: {ports[0]}")

    last: dict[tuple[int, int], int] = {}

    def send_cc(channel: int, cc: int, value: float) -> None:
        midi_value = int(max(0.0, min(1.0, value)) * 127)
        if last.get((channel, cc)) == midi_value:
            return
        last[(channel, cc)] = midi_value
        midi_out.send_message([0xB0 | (channel & 0x0F), cc, midi_value])

    def handle(address: str, *osc_args) -> None:
        if not osc_args:
            return
        value = float(osc_args[0])
        if address == "/global/riser":
            send_cc(0, 11, value)
            return
        prefix, _, param = address.rpartition("/")
        channel = CHANNEL_MAP.get(prefix)
        cc = CC_MAP.get(param)
        if channel is None or cc is None:
            return
        send_cc(channel, cc, value)

    dispatcher = Dispatcher()
    dispatcher.set_default_handler(handle)
    server = ThreadingOSCUDPServer(("127.0.0.1", args.port), dispatcher)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Listening for OSC on 127.0.0.1:{args.port}. Ctrl+C quits.")

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nDone.")
    finally:
        server.shutdown()
        server.server_close()
        midi_out.close_port()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
