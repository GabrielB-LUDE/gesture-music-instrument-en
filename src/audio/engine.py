"""Internal sound engine.

This lets the instrument run without Ableton. A 16th-note step sequencer
plays any number of layers at the same time - one per beat that is
currently held in a hand or placed in the image. Each layer brings its
own preset and its own values:

    gain    -> volume
    cutoff  -> low-pass filter
    density -> which of the four patterns or which loop variant plays

There are two kinds of layers. A layer of kind "steps" builds the beat
from individual hits on the 16th-note grid. A layer of kind "loop" plays
a finished audio file over its full length - these are your own beats
from the `loops/` folder. Both hang on the same clock, so they stay in
sync without any extra work.

The engine is thread-safe: the audio callback only reads a parameter
copy that the video loop sets from outside.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from ..config import AudioConfig
from ..mapping.presets import Preset, preset
from ..vision.types import EngineSnapshot
from . import synth
from .loops import load_library

STEPS_PER_BAR = 16
BARS = 4


@dataclass
class SlotParams:
    preset_id: str
    gain: float = 0.0
    cutoff: float = 0.5
    density: float = 0.0


@dataclass
class Voice:
    buffer: np.ndarray
    position: int = 0
    amp: float = 1.0


@dataclass
class Layer:
    """A sounding layer, belongs to exactly one beat in the image."""

    preset: Preset
    samples: List[np.ndarray]
    filter: synth.OnePoleLowpass
    voices: List[Voice] = field(default_factory=list)
    gain: float = 0.0
    cutoff: float = 0.5
    density: float = 0.0
    current_gain: float = 0.0

    @property
    def is_loop(self) -> bool:
        return self.preset.kind == "loop"

    def trigger(self, index: int, amp: float = 1.0, offset: int = 0) -> None:
        if not self.samples:
            return
        buffer = self.samples[index % len(self.samples)]
        if offset >= len(buffer):
            return
        if len(self.voices) > 12:
            self.voices.pop(0)
        self.voices.append(Voice(buffer=buffer, position=offset, amp=amp))

    def render(self, frames: int, out: np.ndarray) -> None:
        alive: List[Voice] = []
        for voice in self.voices:
            remaining = len(voice.buffer) - voice.position
            if remaining <= 0:
                continue
            take = min(frames, remaining)
            out[:take] += voice.buffer[voice.position:voice.position + take] * voice.amp
            voice.position += take
            if voice.position < len(voice.buffer):
                alive.append(voice)
        self.voices = alive


def pick_sample(
    mode: str, count: int, local: int, bar: int, density: float
) -> Tuple[int, float]:
    """Which sample index and which velocity on this step."""
    accent = 1.0 if local % 4 == 0 else 0.84
    if mode == "hat":
        return (1 if local % 8 == 4 else 0), (0.9 if local % 4 == 0 else 0.6)
    if mode == "roll":
        return 0, (1.0 if local % 4 == 0 else 0.55 + 0.3 * ((local // 2) % 2))
    if mode == "perc":
        return (local // 2) % max(count, 1), 0.9
    if mode == "bass":
        octave = 1 if (density > 0.66 and local % 4 == 2) else 0
        return bar * 2 + octave, 1.0
    if mode == "lead":
        span = 3 + int(density * 4)
        return (local * 3 + bar) % span, 0.9
    if mode == "arp":
        return local + bar, 0.85
    if mode == "bar":
        return bar, 1.0
    if mode == "blip":
        return local // 2 + bar, 0.8
    return 0, accent


class AudioEngine:
    def __init__(self, config: AudioConfig):
        self.config = config
        self.sr = config.samplerate
        self._lock = threading.Lock()
        self._pending: Dict[str, SlotParams] = {}
        self._macro_riser = 0.0
        self._stream = None
        self.running = False

        self.bank = synth.build_sample_bank(self.sr)
        # Your own WAV files override individual bank entries, see
        # src/audio/sample_files.py for adding your own sounds.
        self.bank = synth.apply_custom_samples(self.bank, self.sr)
        # And the finished loops from the `loops/` folder are added as their own
        # banks - one per loop, with one file per density level.
        self.bank.update(load_library(self.sr, config.bpm).bank)
        self.layers: Dict[str, Layer] = {}

        self._samples_per_step = self.sr * 60.0 / (config.bpm * 4.0)
        self._sample_counter = 0.0
        self._next_step = 0

    # ------------------------------------------------------------------
    def update_from_snapshot(self, snapshot: EngineSnapshot) -> None:
        with self._lock:
            self._pending = {
                slot.key: SlotParams(
                    preset_id=slot.preset_id,
                    gain=float(slot.gain),
                    cutoff=float(slot.cutoff),
                    density=float(slot.density),
                )
                for slot in snapshot.slots
            }
            self._macro_riser = float(snapshot.macros.riser)

    def set_slot(self, key: str, preset_id: str, gain: float, cutoff: float, density: float) -> None:
        """Direct access without a snapshot, for demos and tests."""
        with self._lock:
            self._pending[key] = SlotParams(preset_id, gain, cutoff, density)

    # ------------------------------------------------------------------
    def _sync_layers(self) -> None:
        """Transfers the parameter copy into the sounding layers."""
        with self._lock:
            pending = dict(self._pending)

        for key in list(self.layers):
            if key not in pending:
                del self.layers[key]

        for key, params in pending.items():
            layer = self.layers.get(key)
            fresh = layer is None or layer.preset.id != params.preset_id
            if fresh:
                found = preset(params.preset_id)
                layer = Layer(
                    preset=found,
                    samples=self.bank.get(found.bank, [np.zeros(8, dtype=np.float32)]),
                    filter=synth.OnePoleLowpass(self.sr),
                )
                self.layers[key] = layer
            layer.gain = params.gain
            layer.cutoff = params.cutoff
            layer.density = params.density
            if fresh and layer.is_loop:
                # A freshly grabbed loop drops in the middle of the bar.
                # Instead of staying silent until the next bar starts,
                # it enters wherever the clock currently is - it sounds
                # immediately and still sits on the grid.
                self._start_loop(layer, self._loop_offset(layer))

    def _trigger_step(self, step: int) -> None:
        bar = (step // STEPS_PER_BAR) % BARS
        local = step % STEPS_PER_BAR

        for layer in self.layers.values():
            if layer.is_loop:
                # The loop keeps running even while it is quiet:
                # that is the only way it stays in time when the hand turns it up again.
                if step % layer.preset.steps_per_loop == 0:
                    self._start_loop(layer, 0)
                continue
            if layer.gain <= 0.003:
                continue
            if local not in layer.preset.pattern_for(layer.density):
                continue
            index, amp = pick_sample(
                layer.preset.note_mode, len(layer.samples), local, bar, layer.density
            )
            layer.trigger(index, amp=amp)

    def _loop_offset(self, layer: Layer) -> int:
        """Where within its length the loop should currently be."""
        length = layer.preset.steps_per_loop * self._samples_per_step
        if length <= 1.0:
            return 0
        return int(self._sample_counter % length)

    def _start_loop(self, layer: Layer, offset: int) -> None:
        """Trigger the loop file from the start (or from `offset`).

        The density picks the variant, but only ever at the start of a bar -
        cutting over in the middle of a bar would audibly stumble.
        """
        layer.voices.clear()
        layer.trigger(layer.preset.variant_for(layer.density), amp=1.0, offset=offset)

    def _render_block(self, frames: int) -> np.ndarray:
        self._sync_layers()
        mono = np.zeros(frames, dtype=np.float32)
        layer_buffer = np.zeros(frames, dtype=np.float32)

        start = self._sample_counter
        end = start + frames
        while self._next_step * self._samples_per_step < end:
            if self._next_step * self._samples_per_step >= start:
                self._trigger_step(self._next_step)
            self._next_step += 1
        self._sample_counter = end

        with self._lock:
            riser = self._macro_riser

        sounding = 0
        for layer in self.layers.values():
            layer_buffer[:] = 0.0
            layer.render(frames, layer_buffer)
            if not np.any(layer_buffer):
                layer.current_gain += (layer.gain - layer.current_gain) * 0.2
                continue

            sounding += 1
            cutoff_hz = synth.cutoff_to_hz(0.08 + 0.92 * layer.cutoff)
            filtered = layer.filter.process(layer_buffer, cutoff_hz)
            ramp = np.linspace(layer.current_gain, layer.gain, frames, dtype=np.float32)
            layer.current_gain = layer.gain
            mono += filtered * ramp

        # The more layers are running, the further the master level is pulled back.
        mono *= np.float32(1.0 / (1.0 + 0.12 * max(0, sounding - 2)))
        if riser > 0.01:
            mono *= np.float32(1.0 + 0.25 * riser)

        mono = np.tanh(mono * np.float32(self.config.master_gain * 1.15)).astype(np.float32)
        return np.column_stack((mono, mono))

    # ------------------------------------------------------------------
    def render_offline(self, seconds: float) -> np.ndarray:
        """Renders without a sound card, used by the tests."""
        frames = int(seconds * self.sr)
        block = self.config.blocksize
        chunks = []
        done = 0
        while done < frames:
            n = min(block, frames - done)
            chunks.append(self._render_block(n))
            done += n
        return np.concatenate(chunks, axis=0)

    def start(self) -> bool:
        if not self.config.enabled:
            return False
        try:
            import sounddevice as sd
        except Exception as exc:
            print(f"[audio] sounddevice not available ({exc}). Internal engine stays off.")
            return False

        def callback(outdata, frames, time_info, status):  # pragma: no cover
            if status:
                pass
            outdata[:] = self._render_block(frames)

        try:
            self._stream = sd.OutputStream(
                samplerate=self.sr,
                blocksize=self.config.blocksize,
                channels=2,
                dtype="float32",
                callback=callback,
                device=self.config.device,
            )
            self._stream.start()
        except Exception as exc:  # pragma: no cover
            print(f"[audio] Audio output could not be started: {exc}")
            self._stream = None
            return False

        self.running = True
        print(f"[audio] Internal engine running, {self.config.bpm:.0f} BPM")
        return True

    def stop(self) -> None:
        self.running = False
        if self._stream is not None:  # pragma: no cover
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
