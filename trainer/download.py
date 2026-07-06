"""Download augmentation audio and pre-computed negative features.

Everything lands under paths.data:
  mit_rirs/            room impulse responses (reverb augmentation)
  audioset_16k/        background noise clips (AudioSet balanced-train shard)
  fma_16k/             background music clips (Free Music Archive subset)
  negative_datasets/   pre-computed negative spectrogram features published by
                       the microWakeWord author (speech, no_speech,
                       dinner_party, dinner_party_eval)

Note: these datasets carry mixed licenses. Models trained with them should be
treated as suitable for personal, non-commercial use.
"""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from trainer import audio
from trainer.config import SAMPLE_RATE

NEGATIVE_FEATURE_ZIPS = ["speech.zip", "no_speech.zip", "dinner_party.zip", "dinner_party_eval.zip"]
AUDIOSET_SHARD = "bal_train09.tar"


def _write_16k(out_path: Path, clip: np.ndarray) -> None:
    sf.write(str(out_path), np.asarray(clip, dtype=np.float32), SAMPLE_RATE, subtype="PCM_16")


def _extractall(tar: tarfile.TarFile, dest: Path) -> None:
    try:
        tar.extractall(dest, filter="data")
    except TypeError:  # Python < 3.11.4 has no filter parameter
        tar.extractall(dest)


def _convert_dir(src_dir: Path, pattern: str, out_dir: Path, desc: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(src_dir.rglob(pattern))
    for path in tqdm(files, desc=desc):
        out_path = out_dir / (path.stem + ".wav")
        if out_path.exists():
            continue
        try:
            _write_16k(out_path, audio.decode_file(path))
        except Exception as exc:
            tqdm.write(f"SKIPPED {path.name}: {exc}")


def download_mit_rirs(data_dir: Path) -> None:
    out_dir = data_dir / "mit_rirs"
    if out_dir.is_dir() and any(out_dir.glob("*.wav")):
        print("mit_rirs: already downloaded, skipping")
        return
    print("Downloading MIT room impulse responses...")
    import datasets  # deferred: heavy import

    out_dir.mkdir(parents=True, exist_ok=True)
    rir_dataset = datasets.load_dataset(
        "davidscripka/MIT_environmental_impulse_responses", split="train", streaming=True
    )
    for row in tqdm(rir_dataset, desc="MIT RIRs"):
        name = row["audio"]["path"].split("/")[-1]
        _write_16k(out_dir / name, np.asarray(row["audio"]["array"]))


def download_audioset(data_dir: Path) -> None:
    out_dir = data_dir / "audioset_16k"
    if out_dir.is_dir() and any(out_dir.glob("*.wav")):
        print("audioset_16k: already downloaded, skipping")
        return
    print(f"Downloading AudioSet shard {AUDIOSET_SHARD} (~2.5 GB)...")
    tar_path = hf_hub_download(
        repo_id="agkphysics/AudioSet",
        filename=f"data/{AUDIOSET_SHARD}",
        repo_type="dataset",
        local_dir=data_dir / "audioset_raw",
    )
    extract_dir = data_dir / "audioset_raw" / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        _extractall(tar, extract_dir)
    _convert_dir(extract_dir, "*.flac", out_dir, "AudioSet -> 16 kHz wav")


def download_fma(data_dir: Path) -> None:
    out_dir = data_dir / "fma_16k"
    if out_dir.is_dir() and any(out_dir.glob("*.wav")):
        print("fma_16k: already downloaded, skipping")
        return
    print("Downloading Free Music Archive subset (~350 MB)...")
    zip_path = hf_hub_download(
        repo_id="mchl914/fma_xsmall",
        filename="fma_xs.zip",
        repo_type="dataset",
        local_dir=data_dir / "fma_raw",
    )
    extract_dir = data_dir / "fma_raw" / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    _convert_dir(extract_dir, "*.mp3", out_dir, "FMA -> 16 kHz wav")


def download_negative_features(data_dir: Path) -> None:
    out_dir = data_dir / "negative_datasets"
    expected = [out_dir / name.removesuffix(".zip") for name in NEGATIVE_FEATURE_ZIPS]
    if all(path.is_dir() for path in expected):
        print("negative_datasets: already downloaded, skipping")
        return
    print("Downloading pre-computed negative spectrogram features (several GB)...")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in NEGATIVE_FEATURE_ZIPS:
        if (out_dir / name.removesuffix(".zip")).is_dir():
            continue
        zip_path = hf_hub_download(
            repo_id="kahrendt/microwakeword",
            filename=name,
            repo_type="dataset",
            local_dir=data_dir / "negative_raw",
        )
        print(f"Extracting {name}...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir)


def run(cfg: dict) -> None:
    data_dir = Path(cfg["paths"]["data"])
    data_dir.mkdir(parents=True, exist_ok=True)
    download_mit_rirs(data_dir)
    download_audioset(data_dir)
    download_fma(data_dir)
    download_negative_features(data_dir)
    print("All datasets ready.")
