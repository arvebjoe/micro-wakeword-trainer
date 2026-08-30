"""Build augmented spectrogram features from the positive samples.

Combines generated (stage 1) and recorded (ingested) samples, applies audio
augmentation (background noise, reverb, EQ, ...), computes microWakeWord
spectrograms and stores them as memory-mapped ragged arrays under
<paths.training>/features/positives/{training,validation,testing}.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from trainer import ingest
from trainer.config import WINDOW_STEP_MS


def _stage_positives(cfg: dict) -> Path:
    """Collect all positive WAVs into a single directory (Clips reads one dir)."""
    staged = Path(cfg["paths"]["training"]) / "positives" / "all"
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True)

    sources = {
        "generated": Path(cfg["paths"]["generated_samples"]),
        "recorded": ingest.staged_dir(cfg),
    }
    total = 0
    for prefix, src in sources.items():
        if not src.is_dir():
            continue
        for path in sorted(src.glob("*.wav")):
            link = staged / f"{prefix}_{path.name}"
            try:
                link.symlink_to(path.resolve())
            except OSError:
                shutil.copy2(path, link)
            total += 1

    if total < 50:
        sys.exit(
            f"Only {total} positive samples found — that is far too few to train on "
            f"(aim for several hundred). Run `python -m trainer generate` first "
            f"and/or add recordings to {cfg['paths']['recorded_samples']}."
        )
    print(f"Staged {total} positive samples in {staged}")
    return staged


def run(cfg: dict) -> None:
    # Ingest any user recordings first so they are included automatically.
    ingest.run(cfg)
    staged = _stage_positives(cfg)

    data_dir = Path(cfg["paths"]["data"])
    impulse_dir = data_dir / "mit_rirs"
    background_dirs = [data_dir / "audioset_16k", data_dir / "fma_16k"]
    missing = [str(d) for d in [impulse_dir, *background_dirs] if not d.is_dir()]
    if missing:
        sys.exit(
            "Missing augmentation datasets:\n  "
            + "\n  ".join(missing)
            + "\nRun `python -m trainer download` first."
        )

    # Import late: pulls in TensorFlow, which takes a few seconds.
    from microwakeword.audio.augmentation import Augmentation
    from microwakeword.audio.clips import Clips
    from microwakeword.audio.spectrograms import SpectrogramGeneration
    from mmap_ninja.ragged import RaggedMmap

    aug_cfg = cfg["augmentation"]
    clips = Clips(
        input_directory=str(staged),
        file_pattern="*.wav",
        max_clip_duration_s=None,
        remove_silence=bool(aug_cfg["remove_silence"]),
        random_split_seed=10,
        split_count=0.1,
    )
    augmenter = Augmentation(
        augmentation_duration_s=float(aug_cfg["duration_s"]),
        augmentation_probabilities=dict(aug_cfg["probabilities"]),
        impulse_paths=[str(impulse_dir)],
        background_paths=[str(d) for d in background_dirs],
        background_min_snr_db=int(aug_cfg["background_min_snr_db"]),
        background_max_snr_db=int(aug_cfg["background_max_snr_db"]),
        min_jitter_s=0.195,
        max_jitter_s=0.205,
    )

    features_root = Path(cfg["paths"]["training"]) / "features" / "positives"
    # (split name in Clips, output folder, repetitions, slide_frames)
    # slide_frames > 1 shifts each spectrogram to simulate streaming inference;
    # the testing set runs the actual streaming model, so it uses 1.
    plan = [
        ("train", "training", int(aug_cfg["training_repetitions"]), 10),
        ("validation", "validation", 1, 10),
        ("test", "testing", 1, 1),
    ]
    for split, folder, repetition, slide_frames in plan:
        out_dir = features_root / folder
        if (out_dir / "wakeword_mmap").is_dir():
            print(f"{folder}: features already exist, skipping (delete {out_dir} to rebuild)")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        spectrograms = SpectrogramGeneration(
            clips=clips,
            augmenter=augmenter,
            slide_frames=slide_frames,
            step_ms=WINDOW_STEP_MS,
        )
        print(f"Generating {folder} features ({repetition}x augmented)...")
        RaggedMmap.from_generator(
            out_dir=str(out_dir / "wakeword_mmap"),
            sample_generator=spectrograms.spectrogram_generator(
                split=split, repeat=repetition
            ),
            batch_size=100,
            verbose=True,
        )

    print(f"Features written to {features_root}")
