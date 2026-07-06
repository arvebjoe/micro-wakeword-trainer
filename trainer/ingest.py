"""Normalize user-recorded wake word clips to the training format.

Reads every audio file in paths.recorded_samples (wav, mp3, flac, ogg, m4a, ...)
and writes 16 kHz mono 16-bit WAVs into <paths.training>/positives/recorded/.
Stage 2 trains on generated and recorded samples together.
"""

from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from trainer import audio


def staged_dir(cfg: dict) -> Path:
    return Path(cfg["paths"]["training"]) / "positives" / "recorded"


def run(cfg: dict) -> int:
    src = Path(cfg["paths"]["recorded_samples"])
    dst = staged_dir(cfg)
    dst.mkdir(parents=True, exist_ok=True)

    suffixes = audio.SOUNDFILE_SUFFIXES | audio.FFMPEG_SUFFIXES
    files = sorted(
        p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in suffixes
    )
    if not files:
        print(f"No recordings found in {src} — skipping.")
        return 0

    converted = 0
    for path in tqdm(files, desc="Ingesting recordings"):
        out_path = dst / f"{path.stem}.wav"
        if out_path.exists() and out_path.stat().st_mtime >= path.stat().st_mtime:
            converted += 1
            continue
        try:
            clip = audio.peak_normalize(audio.decode_file(path))
        except Exception as exc:
            tqdm.write(f"SKIPPED {path.name}: {exc}")
            continue
        audio.write_wav(out_path, clip)
        converted += 1

    print(f"Ingested {converted}/{len(files)} recordings into {dst}")
    return converted
