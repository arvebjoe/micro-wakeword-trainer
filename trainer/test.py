"""Test a trained wake word model on WAV files or the live microphone.

File mode scores each clip and reports whether it would trigger. Mic mode
replicates the ESPHome detection logic in real time: the streaming model runs
every 30 ms and a detection fires when the moving average of the last
`sliding_window_size` probabilities exceeds `probability_cutoff` (both read
from the exported manifest, overridable with --cutoff).
"""

from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

from trainer import audio
from trainer.config import SAMPLE_RATE

REFRACTORY_S = 2.0  # ignore re-triggers for this long after a detection


class Detector:
    """Moving-average threshold over per-step probabilities (ESPHome logic)."""

    def __init__(self, cutoff: float, window_size: int):
        self.cutoff = cutoff
        self.window = deque(maxlen=window_size)

    def update(self, probability: float) -> tuple[float, bool]:
        self.window.append(probability)
        average = sum(self.window) / self.window.maxlen
        return average, average > self.cutoff

    def reset(self) -> None:
        self.window.clear()


def _load_model(cfg: dict, cutoff_override: float | None):
    from microwakeword.inference import Model  # deferred: imports TensorFlow deps

    output_dir = Path(cfg["paths"]["output"])
    tflite_path = output_dir / f"{cfg['model_name']}.tflite"
    manifest_path = output_dir / f"{cfg['model_name']}.json"
    if not tflite_path.is_file():
        sys.exit(f"No model at {tflite_path} — run `python -m trainer train` first.")

    cutoff = cfg["manifest"]["probability_cutoff"]
    window_size = cfg["manifest"]["sliding_window_size"]
    if manifest_path.is_file():
        micro = json.loads(manifest_path.read_text())["micro"]
        cutoff = micro["probability_cutoff"]
        window_size = micro["sliding_window_size"]
    if cutoff_override is not None:
        cutoff = cutoff_override

    model = Model(str(tflite_path))
    print(f"Loaded {tflite_path.name} (cutoff={cutoff}, window={window_size})")
    return model, Detector(float(cutoff), int(window_size))


def test_files(cfg: dict, paths: list[str], cutoff_override: float | None = None) -> bool:
    model, detector = _load_model(cfg, cutoff_override)

    files: list[Path] = []
    for entry in paths:
        p = Path(entry)
        if p.is_dir():
            files.extend(sorted(q for q in p.rglob("*") if q.suffix.lower() in audio.SOUNDFILE_SUFFIXES | audio.FFMPEG_SUFFIXES))
        elif p.is_file():
            files.append(p)
        else:
            sys.exit(f"no such file or directory: {p}")
    if not files:
        sys.exit("no audio files found to test")

    detected_count = 0
    for path in files:
        clip = audio.decode_file(path)
        probabilities = model.predict_clip(clip)

        detector.reset()
        peak = 0.0
        detected = False
        for probability in probabilities:
            average, fired = detector.update(float(probability))
            peak = max(peak, average)
            detected = detected or fired

        detected_count += detected
        marker = "DETECTED" if detected else "-"
        print(f"{marker:>8}  peak={peak:.3f}  {path}")

    print(f"\n{detected_count}/{len(files)} clips detected "
          f"(peak = highest moving-average probability; cutoff {detector.cutoff})")
    return detected_count > 0


def test_microphone(cfg: dict, cutoff_override: float | None = None) -> None:
    try:
        import sounddevice as sd
    except OSError as exc:
        sys.exit(
            f"sounddevice could not load PortAudio ({exc}).\n"
            "On Debian/Ubuntu: sudo apt install libportaudio2"
        )

    from pymicro_features import MicroFrontend

    model, detector = _load_model(cfg, cutoff_override)
    stride = int(model.stride)  # feature frames per inference (10 ms each)

    frontend = MicroFrontend()
    pending_audio = b""
    pending_frames: list[list[int]] = []
    refractory_steps_left = 0
    refractory_steps = int(REFRACTORY_S * 1000 / (10 * stride))
    chunk_bytes = 160 * 2  # 10 ms of 16-bit samples

    print("Listening... say the wake word (Ctrl+C to stop)")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16") as stream:
        while True:
            block, _ = stream.read(160)
            pending_audio += block.tobytes()

            while len(pending_audio) >= chunk_bytes:
                result = frontend.process_samples(pending_audio[:chunk_bytes])
                pending_audio = pending_audio[result.samples_read * 2 :]
                if result.features:
                    pending_frames.append(result.features)

            while len(pending_frames) >= stride:
                frames = np.array(pending_frames[:stride], dtype=np.float32)
                pending_frames = pending_frames[stride:]
                probability = float(model.predict_spectrogram(frames)[0])
                average, fired = detector.update(probability)

                if refractory_steps_left > 0:
                    refractory_steps_left -= 1
                    continue
                bar = "#" * int(average * 40)
                print(f"\r{average:5.3f} |{bar:<40}|", end="", flush=True)
                if fired:
                    print(f"\n*** DETECTED (avg={average:.3f}) ***")
                    detector.reset()
                    refractory_steps_left = refractory_steps
