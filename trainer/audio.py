"""Audio helpers: decode, resample to 16 kHz mono, normalize, write WAV."""

from __future__ import annotations

import io
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from trainer.config import SAMPLE_RATE

# Formats libsndfile decodes natively (mp3 requires libsndfile >= 1.1).
SOUNDFILE_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif"}
# Anything else (m4a/aac/opus-in-mp4/...) goes through ffmpeg when available.
FFMPEG_SUFFIXES = {".m4a", ".aac", ".mp4", ".opus", ".wma", ".webm"}


def to_mono_16k(audio: np.ndarray, rate: int) -> np.ndarray:
    """Convert float audio of any rate/channels to 16 kHz mono float32."""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        audio = soxr.resample(audio, rate, SAMPLE_RATE)
    return np.asarray(audio, dtype=np.float32)


def peak_normalize(audio: np.ndarray, peak: float = 0.8) -> np.ndarray:
    max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_abs > 1e-6:
        audio = audio * (peak / max_abs)
    return audio


def speed_perturb(audio: np.ndarray, factor: float) -> np.ndarray:
    """Change playback speed (and pitch) by resampling. factor > 1 = faster."""
    if abs(factor - 1.0) < 1e-3:
        return audio
    stretched = soxr.resample(audio, SAMPLE_RATE, int(round(SAMPLE_RATE / factor)))
    return np.asarray(stretched, dtype=np.float32)


def write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, SAMPLE_RATE, subtype="PCM_16")


def decode_bytes(data: bytes) -> np.ndarray:
    """Decode an in-memory audio file (e.g. a WAV HTTP response) to 16 kHz mono."""
    audio, rate = sf.read(io.BytesIO(data), dtype="float32")
    return to_mono_16k(audio, rate)


def decode_file(path: Path) -> np.ndarray:
    """Decode an audio file from disk to 16 kHz mono float32.

    Uses libsndfile for common formats and falls back to ffmpeg for the rest.
    """
    suffix = path.suffix.lower()
    if suffix in SOUNDFILE_SUFFIXES:
        try:
            audio, rate = sf.read(str(path), dtype="float32")
            return to_mono_16k(audio, rate)
        except sf.LibsndfileError:
            pass  # e.g. mp3 on an old libsndfile — try ffmpeg below

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            f"cannot decode {path.name}: unsupported by libsndfile and ffmpeg is not "
            f"installed. Install ffmpeg or convert the file to WAV/FLAC first."
        )
    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path),
            "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-",
        ],
        capture_output=True,
        check=True,
    )
    return np.frombuffer(result.stdout, dtype=np.float32).copy()
