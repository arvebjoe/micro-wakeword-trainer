"""Stage 2: train the model and export .tflite + ESPHome JSON manifest."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from trainer.config import WINDOW_STEP_MS, validate_model_geometry

TFLITE_SUBPATH = Path("tflite_stream_state_internal_quant") / "stream_state_internal_quant.tflite"


def _training_yaml(cfg: dict, train_dir: Path) -> dict:
    training = cfg["training"]
    features_root = Path(cfg["paths"]["training"]) / "features" / "positives"
    negatives = Path(cfg["paths"]["data"]) / "negative_datasets"

    def feature_set(directory, weight, truth, truncation):
        return {
            "features_dir": str(directory),
            "sampling_weight": weight,
            "penalty_weight": 1.0,
            "truth": truth,
            "truncation_strategy": truncation,
            "type": "mmap",
        }

    return {
        "window_step_ms": WINDOW_STEP_MS,
        "train_dir": str(train_dir),
        "features": [
            feature_set(features_root, 2.0, True, "truncate_start"),
            feature_set(negatives / "speech", 10.0, False, "random"),
            feature_set(negatives / "dinner_party", 10.0, False, "random"),
            feature_set(negatives / "no_speech", 5.0, False, "random"),
            # sampling_weight 0: only used for the *_ambient validation/testing
            # sets that estimate false accepts per hour.
            feature_set(negatives / "dinner_party_eval", 0.0, False, "split"),
        ],
        "training_steps": list(training["steps"]),
        "positive_class_weight": list(training["positive_class_weight"]),
        "negative_class_weight": list(training["negative_class_weight"]),
        "learning_rates": list(training["learning_rates"]),
        "batch_size": int(training["batch_size"]),
        "time_mask_max_size": list(training["time_mask_max_size"]),
        "time_mask_count": list(training["time_mask_count"]),
        "freq_mask_max_size": list(training["freq_mask_max_size"]),
        "freq_mask_count": list(training["freq_mask_count"]),
        "eval_step_interval": int(training["eval_step_interval"]),
        "clip_duration_ms": int(training["clip_duration_ms"]),
        "target_minimization": 0.9,
        "minimization_metric": None,
        "maximization_metric": training["maximization_metric"],
    }


def _check_inputs(cfg: dict) -> None:
    features_root = Path(cfg["paths"]["training"]) / "features" / "positives"
    if not (features_root / "training").is_dir():
        sys.exit(f"No positive features at {features_root} — run `python -m trainer features` first.")
    negatives = Path(cfg["paths"]["data"]) / "negative_datasets"
    for name in ["speech", "no_speech", "dinner_party", "dinner_party_eval"]:
        if not (negatives / name).is_dir():
            sys.exit(f"Missing negative dataset {negatives / name} — run `python -m trainer download` first.")


def write_manifest(cfg: dict, tflite_name: str, manifest_path: Path) -> None:
    m = cfg["manifest"]
    manifest = {
        "type": "micro",
        "wake_word": cfg["wake_word"],
        "author": cfg["author"],
        "website": cfg["website"],
        "model": tflite_name,
        "trained_languages": [cfg["language"]],
        "version": 2,
        "micro": {
            "probability_cutoff": float(m["probability_cutoff"]),
            "feature_step_size": WINDOW_STEP_MS,
            "sliding_window_size": int(m["sliding_window_size"]),
            "tensor_arena_size": int(m["tensor_arena_size"]),
            "minimum_esphome_version": str(m["minimum_esphome_version"]),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def run(cfg: dict, train_flag: bool = True) -> None:
    validate_model_geometry(cfg)
    _check_inputs(cfg)

    train_dir = Path(cfg["paths"]["training"]) / "trained" / cfg["model_name"]
    train_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(cfg["paths"]["training"]) / "training_parameters.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(_training_yaml(cfg, train_dir), f)

    model = cfg["model"]
    cmd = [
        sys.executable, "-m", "microwakeword.model_train_eval",
        f"--training_config={config_path}",
        "--train", "1" if train_flag else "0",
        "--restore_checkpoint", "1",
        "--test_tf_nonstreaming", "0",
        "--test_tflite_nonstreaming", "0",
        "--test_tflite_nonstreaming_quantized", "0",
        "--test_tflite_streaming", "0",
        "--test_tflite_streaming_quantized", "1",
        "--use_weights", "best_weights",
        "mixednet",
        "--pointwise_filters", str(model["pointwise_filters"]),
        "--repeat_in_block", str(model["repeat_in_block"]),
        "--mixconv_kernel_sizes", str(model["mixconv_kernel_sizes"]),
        "--residual_connection", str(model["residual_connection"]),
        "--first_conv_filters", str(model["first_conv_filters"]),
        "--first_conv_kernel_size", str(model["first_conv_kernel_size"]),
        "--stride", str(model["stride"]),
    ]
    print("Starting microWakeWord training:\n  " + " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"training failed with exit code {result.returncode}")

    tflite_src = train_dir / TFLITE_SUBPATH
    if not tflite_src.is_file() or tflite_src.stat().st_size == 0:
        sys.exit(f"training finished but no model was produced at {tflite_src}")

    output_dir = Path(cfg["paths"]["output"])
    output_dir.mkdir(parents=True, exist_ok=True)
    tflite_name = f"{cfg['model_name']}.tflite"
    shutil.copy2(tflite_src, output_dir / tflite_name)
    write_manifest(cfg, tflite_name, output_dir / f"{cfg['model_name']}.json")

    roc_path = tflite_src.parent / "tflite_streaming_roc.txt"
    print("\n" + "=" * 60)
    print(f"Model:    {output_dir / tflite_name}")
    print(f"Manifest: {output_dir / (cfg['model_name'] + '.json')}")
    if roc_path.is_file():
        print("\nCutoff tradeoffs measured on the test sets")
        print("(frr = false reject rate, faph = false accepts/hour —")
        print(f" set manifest.probability_cutoff in config.yaml accordingly):\n")
        print(roc_path.read_text())
    print("=" * 60)
