"""Application loop: lobby, countdown, playing."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

from .audio.engine import AudioEngine
from .audio.loops import load_library
from .config import Config
from .mapping import presets as preset_library
from .mapping.presets import PRESETS
from .engine import InstrumentEngine
from .network.osc_bridge import build_bridge
from .vision.camera import Camera, CameraError, SyntheticCamera
from .vision.tracker import build_tracker

KEY_ESC = 27
STATE_LOBBY = "lobby"
STATE_COUNTDOWN = "countdown"
STATE_PLAY = "play"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gesture-music-instrument",
        description="Gesture-controlled multiplayer musical instrument",
    )
    parser.add_argument("--config", default="config.json", help="Path to the JSON configuration")
    parser.add_argument("--performers", type=int, choices=(1, 2, 3), help="Number of players")
    parser.add_argument("--skip-lobby", action="store_true", help="start without the start screen")
    parser.add_argument("--countdown", type=float, help="Seconds between selection and start")
    parser.add_argument("--camera", type=int, help="Camera index (0, 1, 2 ...)")
    parser.add_argument("--width", type=int, help="Camera width")
    parser.add_argument("--height", type=int, help="Camera height")
    parser.add_argument("--bpm", type=float, help="Tempo of the internal engine")
    parser.add_argument("--osc-port", type=int, help="OSC target port")
    parser.add_argument("--osc-host", help="OSC target host")
    parser.add_argument("--no-osc", action="store_true", help="Disable OSC output")
    parser.add_argument("--no-audio", action="store_true", help="internal sound engine off")
    parser.add_argument(
        "--library", choices=("auto", "loops", "builtin"),
        help="where the beats come from: your own loops from loops/, the built-in ones, or auto",
    )
    parser.add_argument("--no-window", action="store_true", help="run without a display window")
    parser.add_argument("--fullscreen", action="store_true", help="start the window in fullscreen")
    parser.add_argument("--no-mirror", action="store_true", help="do not mirror the image")
    parser.add_argument("--mock", action="store_true", help="virtual hands instead of a camera")
    parser.add_argument("--seed", type=int, help="fixed random seed for the glove positions")
    parser.add_argument("--frames", type=int, help="stop after N frames (test run)")
    parser.add_argument("--preview", help="save the last frame as PNG")
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    config = Config.load(args.config)
    if args.performers:
        config.performers = args.performers
    if args.camera is not None:
        config.camera.device = args.camera
    if args.width:
        config.camera.width = args.width
    if args.height:
        config.camera.height = args.height
    if args.bpm:
        config.audio.bpm = args.bpm
    if args.osc_port:
        config.osc.port = args.osc_port
    if args.osc_host:
        config.osc.host = args.osc_host
    if args.no_osc:
        config.osc.enabled = False
    if args.no_audio:
        config.audio.enabled = False
    if args.library:
        config.audio.library = args.library
    if args.no_window:
        config.ui.show_window = False
    if args.fullscreen:
        config.ui.fullscreen = True
    if args.no_mirror:
        config.camera.mirror = False
    if args.countdown is not None:
        config.ui.countdown_seconds = args.countdown
    return config


def install_library(config: Config) -> str:
    """Decides which beats sit in the library.

    If the `loops/` folder contains at least one WAV file, it wins: that
    is the way to bring in your own loops without touching a single line
    of code (instructions in `src/audio/loops.py` and `loops/README.md`).
    Must run before the engine is built, because the forge remembers the
    library when it is created.
    """
    mode = (config.audio.library or "auto").lower()
    if mode == "builtin":
        preset_library.reset_presets()
        return f"{len(PRESETS)} built-in beats"

    found = load_library(config.audio.samplerate, config.audio.bpm)
    for problem in found.problems:
        print(f"[loops] skipped - {problem}")
    if found.presets:
        preset_library.use_presets(found.presets)
        variants = sum(item.variants for item in found.presets)
        return f"{len(found.presets)} own loops ({variants} files) from loops/"
    if mode == "loops":
        print("[loops] no own loops found, sticking with the built-in ones")
    preset_library.reset_presets()
    return f"{len(PRESETS)} built-in beats"


def run(args: Optional[argparse.Namespace] = None) -> int:
    args = args or build_parser().parse_args()
    config = config_from_args(args)
    library_note = install_library(config)

    camera = SyntheticCamera(config.camera) if args.mock else Camera(config.camera)
    try:
        camera.open()
    except CameraError as exc:
        print(f"[error] {exc}")
        return 2

    try:
        tracker = build_tracker(
            config.tracking, mock=args.mock, performers=config.performers
        )
    except Exception as exc:
        camera.close()
        print(f"[error] Tracker could not be started: {exc}")
        return 3

    engine = InstrumentEngine(config, seed=args.seed)
    bridge = build_bridge(config.osc)
    audio = AudioEngine(config.audio)
    audio_ok = audio.start()

    want_canvas = config.ui.show_window or bool(args.preview)
    display = lobby = None
    if want_canvas:
        from .ui.display import Display
        from .ui.lobby import Lobby, countdown_overlay

        display = Display(config)
        lobby = Lobby(config)

    state = STATE_PLAY
    if want_canvas and not args.skip_lobby:
        state = STATE_LOBBY
    countdown_until = 0.0

    print(
        f"[start] {library_note} | "
        f"OSC {'on' if config.osc.enabled else 'off'} "
        f"({config.osc.host}:{config.osc.port}) | "
        f"internal engine {'on' if audio_ok else 'off'}"
    )
    print(
        "[keys] ESC quit | f fullscreen | m mirroring | h skeleton | "
        "r shuffle library | space put everything on the rack | x delete everything | "
        "l back to selection"
    )

    last_canvas = None
    frames = 0
    started = time.time()

    try:
        while True:
            frame = camera.read()
            if frame is None:
                time.sleep(0.003)
                continue

            observations = tracker.process(frame)
            frames += 1

            # ---------------- Lobby ----------------
            if state == STATE_LOBBY:
                choice = lobby.update(observations)
                last_canvas = lobby.render(frame.image)
                if config.ui.show_window:
                    key = display.show(last_canvas)
                    if key in (KEY_ESC, ord("q")):
                        break
                    if key != 255:
                        choice = lobby.handle_key(key) or choice
                if choice:
                    engine.set_performers(choice)
                    lobby.reset()
                    countdown_until = time.time() + config.ui.countdown_seconds
                    state = STATE_COUNTDOWN
                    print(f"[info] {choice} players, {len(PRESETS)} beats in the library")
                if args.frames and frames >= args.frames:
                    break
                continue

            # ---------------- Countdown ----------------
            if state == STATE_COUNTDOWN:
                remaining = countdown_until - time.time()
                snapshot = engine.update(observations)
                base = display.render(frame.image, snapshot, "START")
                last_canvas = countdown_overlay(base, remaining)
                if config.ui.show_window:
                    key = display.show(last_canvas)
                    if key in (KEY_ESC, ord("q")):
                        break
                if remaining <= 0:
                    state = STATE_PLAY
                if args.frames and frames >= args.frames:
                    break
                continue

            # ---------------- Playing ----------------
            snapshot = engine.update(observations)
            bridge.publish(snapshot)
            if audio_ok:
                audio.update_from_snapshot(snapshot)

            if display is not None:
                status = "OSC" if config.osc.enabled else ""
                status += " AUDIO" if audio_ok else ""
                last_canvas = display.render(frame.image, snapshot, status.strip())

            if display is not None and config.ui.show_window:
                key = display.show(last_canvas)
                if key in (KEY_ESC, ord("q")):
                    break
                if key == ord("f"):
                    display.toggle_fullscreen()
                elif key == ord("m"):
                    config.camera.mirror = not config.camera.mirror
                    camera.config.mirror = config.camera.mirror
                elif key == ord("h"):
                    config.ui.draw_landmarks = not config.ui.draw_landmarks
                elif key == ord("r"):
                    engine.shuffle_library()
                    print("[info] library reshuffled")
                elif key == ord(" "):
                    engine.place_all()
                    print("[info] all loops put on the rack")
                elif key == ord("x"):
                    engine.clear()
                    print("[info] everything deleted")
                elif key == ord("l") and lobby is not None:
                    engine.clear()
                    state = STATE_LOBBY

            if args.frames and frames >= args.frames:
                break
    except KeyboardInterrupt:
        print("\n[stop] Cancelled by user")
    finally:
        if args.preview and last_canvas is not None:
            from .ui.display import save_preview

            save_preview(last_canvas, args.preview)
            print(f"[info] Preview saved: {os.path.abspath(args.preview)}")

        duration = max(time.time() - started, 1e-6)
        print(f"[stop] {frames} frames in {duration:.1f} s ({frames / duration:.1f} FPS)")
        tracker.close()
        camera.close()
        bridge.close()
        audio.stop()
        if display is not None:
            display.close()
    return 0


def main() -> int:
    try:
        return run()
    except Exception as exc:  # pragma: no cover
        print(f"[error] {exc}", file=sys.stderr)
        return 1
