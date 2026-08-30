"""Load and validate the YAML configuration."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = "config.yaml"

SAMPLE_RATE = 16000  # Hz, fixed by the microWakeWord feature preprocessor
WINDOW_STEP_MS = 10  # spectrogram feature step used across the pipeline


class ConfigError(Exception):
    pass


def _require(cfg: dict, key: str) -> Any:
    if key not in cfg or cfg[key] in (None, ""):
        raise ConfigError(f"config is missing required key: {key!r}")
    return cfg[key]


def slugify(text: str) -> str:
    """'Hey Computer!' -> 'hey_computer' (used for file/model names)."""
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    if not slug:
        raise ConfigError(f"cannot derive a model name from wake word {text!r}")
    return slug


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(
            f"config file not found: {path} (copy/edit config.yaml in the repo root)"
        )
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    _require(cfg, "wake_word")

    cfg.setdefault("phonetic_spellings", [])
    cfg.setdefault("author", "")
    cfg.setdefault("website", "")
    cfg.setdefault("language", "en")
    cfg["model_name"] = cfg.get("model_name") or slugify(cfg["wake_word"])

    paths = cfg.setdefault("paths", {})
    paths.setdefault("generated_samples", "samples/generated")
    paths.setdefault("recorded_samples", "samples/recorded")
    paths.setdefault("data", "data")
    paths.setdefault("training", "training")
    paths.setdefault("output", "models")
    # Resolve all paths relative to the config file's directory so the CLI
    # works from anywhere.
    root = path.parent.resolve()
    for key, value in paths.items():
        paths[key] = str((root / value).resolve())

    openai_cfg = cfg.setdefault("openai", {})
    openai_cfg.setdefault("model", "gpt-4o-mini-tts")
    openai_cfg.setdefault("num_samples", 400)
    openai_cfg.setdefault(
        "voices",
        ["alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse"],
    )
    openai_cfg.setdefault("instructions", ["Speak naturally at a normal conversational pace."])
    openai_cfg.setdefault("speed_range", [0.85, 1.2])
    openai_cfg.setdefault("concurrency", 4)

    aug = cfg.setdefault("augmentation", {})
    aug.setdefault("duration_s", 3.2)
    aug.setdefault(
        "probabilities",
        {
            "SevenBandParametricEQ": 0.1,
            "TanhDistortion": 0.1,
            "PitchShift": 0.1,
            "BandStopFilter": 0.1,
            "AddColorNoise": 0.1,
            "AddBackgroundNoise": 0.75,
            "Gain": 1.0,
            "RIR": 0.5,
        },
    )
    aug.setdefault("background_min_snr_db", -5)
    aug.setdefault("background_max_snr_db", 10)
    aug.setdefault("remove_silence", False)
    aug.setdefault("training_repetitions", 2)

    training = cfg.setdefault("training", {})
    training.setdefault("steps", [20000])
    training.setdefault("learning_rates", [0.001])
    training.setdefault("batch_size", 128)
    training.setdefault("positive_class_weight", [1])
    training.setdefault("negative_class_weight", [20])
    training.setdefault("time_mask_max_size", [0])
    training.setdefault("time_mask_count", [0])
    training.setdefault("freq_mask_max_size", [0])
    training.setdefault("freq_mask_count", [0])
    training.setdefault("eval_step_interval", 500)
    training.setdefault("clip_duration_ms", 1500)
    training.setdefault("maximization_metric", "average_viable_recall")

    model = cfg.setdefault("model", {})
    model.setdefault("pointwise_filters", "64,64,64,64")
    model.setdefault("repeat_in_block", "1,1,1,1")
    model.setdefault("mixconv_kernel_sizes", "[5],[7,11],[9,15],[23]")
    model.setdefault("residual_connection", "0,0,0,0")
    model.setdefault("first_conv_filters", 32)
    model.setdefault("first_conv_kernel_size", 5)
    model.setdefault("stride", 3)

    manifest = cfg.setdefault("manifest", {})
    manifest.setdefault("probability_cutoff", 0.97)
    manifest.setdefault("sliding_window_size", 5)
    manifest.setdefault("tensor_arena_size", 30000)
    manifest.setdefault("minimum_esphome_version", "2024.7.0")

    return cfg


def spectrogram_length(cfg: dict) -> tuple[int, int]:
    """Compute the model input spectrogram length the same way
    microwakeword.model_train_eval does, and return (length, final_layer_length).

    The quantization step requires length % stride == 0; validating here fails
    fast with an actionable message instead of crashing after hours of training.
    """
    model = cfg["model"]
    stride = int(model["stride"])
    clip_duration_ms = int(cfg["training"]["clip_duration_ms"])

    desired_samples = int(SAMPLE_RATE * clip_duration_ms / 1000)
    window_size_samples = int(SAMPLE_RATE * 30 / 1000)
    window_step_samples = int(stride * SAMPLE_RATE * WINDOW_STEP_MS / 1000)
    length_minus_window = desired_samples - window_size_samples
    final_layer = 0 if length_minus_window < 0 else 1 + length_minus_window // window_step_samples

    # microwakeword.mixednet.spectrogram_slices_dropped
    dropped = 0
    if int(model["first_conv_filters"]) > 0:
        dropped += int(model["first_conv_kernel_size"]) - 1
    repeats = [int(x) for x in str(model["repeat_in_block"]).split(",")]
    kernel_groups = re.findall(r"\[([^\]]*)\]", str(model["mixconv_kernel_sizes"]))
    kernels = [max(int(k) for k in group.split(",")) for group in kernel_groups]
    if len(repeats) != len(kernels):
        raise ConfigError(
            "model.repeat_in_block and model.mixconv_kernel_sizes must have the same number of blocks"
        )
    for repeat, ksize in zip(repeats, kernels):
        dropped += repeat * (ksize - 1) * stride

    return final_layer + dropped, final_layer


def validate_model_geometry(cfg: dict) -> None:
    length, _ = spectrogram_length(cfg)
    stride = int(cfg["model"]["stride"])
    if length % stride != 0:
        raise ConfigError(
            f"invalid model geometry: the spectrogram length ({length}) must be divisible "
            f"by model.stride ({stride}), or post-training quantization will fail. "
            f"Adjust model.first_conv_kernel_size (currently "
            f"{cfg['model']['first_conv_kernel_size']}) or training.clip_duration_ms "
            f"(currently {cfg['training']['clip_duration_ms']}). The defaults "
            f"(kernel size 5, clip duration 1500, stride 3) are known-good."
        )
