"""Command line interface: python -m trainer <command>."""

from __future__ import annotations

import argparse

from trainer.config import DEFAULT_CONFIG_PATH, load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="trainer",
        description="Train a microWakeWord wake word model for ESPHome / Home Assistant.",
        epilog=(
            "Typical run: generate -> download -> features -> train. "
            "Or `python -m trainer all` to run everything."
        ),
    )
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG_PATH, help="path to config.yaml"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_generate = sub.add_parser(
        "generate", help="stage 1: synthesize wake word samples with OpenAI TTS"
    )
    p_generate.add_argument(
        "--dry-run", action="store_true", help="show what would be generated, no API calls"
    )
    sub.add_parser("ingest", help="convert user recordings to 16 kHz training WAVs")
    sub.add_parser("download", help="fetch augmentation + negative datasets (several GB)")
    sub.add_parser("features", help="build augmented spectrogram features")
    p_train = sub.add_parser("train", help="stage 2: train and export .tflite + .json")
    p_train.add_argument(
        "--export-only",
        action="store_true",
        help="skip training; convert/test the current best checkpoint and export",
    )
    sub.add_parser("all", help="run download, generate, features and train in order")

    args = parser.parse_args()
    cfg = load_config(args.config)

    # Imports are deferred: several commands pull in TensorFlow or make
    # network connections, and `--help` should stay instant.
    if args.command == "generate":
        from trainer import generate

        generate.run(cfg, dry_run=args.dry_run)
    elif args.command == "ingest":
        from trainer import ingest

        ingest.run(cfg)
    elif args.command == "download":
        from trainer import download

        download.run(cfg)
    elif args.command == "features":
        from trainer import features

        features.run(cfg)
    elif args.command == "train":
        from trainer import train

        train.run(cfg, train_flag=not args.export_only)
    elif args.command == "all":
        from trainer import download, features, generate, train

        generate.run(cfg)
        download.run(cfg)
        features.run(cfg)
        train.run(cfg)


if __name__ == "__main__":
    main()
