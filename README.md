# micro-wakeword-trainer

Train your own **micro wake word** model for ESP32 voice assistants in the
Home Assistant ecosystem — fully locally, accelerated by an NVIDIA GPU.

The pipeline produces the two files the
[ESPHome `micro_wake_word` component](https://esphome.io/components/micro_wake_word/)
needs:

- `models/<wake_word>.tflite` — quantized streaming model
- `models/<wake_word>.json` — v2 model manifest

Under the hood it drives [microWakeWord](https://github.com/OHF-Voice/micro-wake-word),
the official training framework by Kevin Ahrendt used for the published
"Okay Nabu" / "Hey Jarvis" / "Alexa" models.

## How it works

**Stage 1 — collect voice samples** (`generate` + your own recordings)

Hundreds of short clips of the wake word are synthesized with the OpenAI TTS
API (`gpt-4o-mini-tts`), rotating through 11 voices and multiple delivery
styles (fast, slow, soft, loud, ...), with random speed perturbation for extra
variety. Clips are written as 16 kHz mono WAVs to `samples/generated/`.

You can (and should!) also record yourself and family members saying the wake
word and drop those files into `samples/recorded/` — any common format works
(wav/mp3/flac/ogg/m4a). Real voices noticeably improve the model for the
people who actually use it.

**Stage 2 — train the model** (`download` + `features` + `train`)

1. `download` fetches augmentation data (MIT room impulse responses, AudioSet
   background noise, Free Music Archive music) and ~10 GB of pre-computed
   *negative* spectrogram features (general speech, ambient noise, "dinner
   party" babble) published by the microWakeWord author.
2. `features` augments your positive samples — random background noise,
   reverb, EQ, distortion, pitch shift — and computes spectrogram features.
3. `train` trains a streaming MixedNet classifier, quantizes it to int8,
   converts it to a streaming TFLite model, measures false-reject /
   false-accept tradeoffs, and writes the `.tflite` + `.json` pair to
   `models/`.

## Requirements

- Linux (or WSL2) with **Python 3.10–3.12**
- **NVIDIA GPU** strongly recommended (CPU works but is ~10x slower).
  The pinned `tensorflow[and-cuda]` wheel bundles CUDA/cuDNN — you only need
  a recent NVIDIA driver, no CUDA toolkit install.
- An **OpenAI API key** for stage 1 (~$0.02 per 100 samples with
  `gpt-4o-mini-tts`)
- ~20 GB free disk space (datasets + features)

## Quick start

```bash
./setup.sh                        # creates .venv and installs everything
source .venv/bin/activate

# 1. Configure
$EDITOR config.yaml               # set wake_word, author, ...

# 2. Stage 1: voice samples
export OPENAI_API_KEY=sk-...
python -m trainer generate       # ~400 TTS samples by default
#    ...and optionally record yourself and copy files into samples/recorded/

# 3. Stage 2: train
python -m trainer download       # one-time, several GB
python -m trainer features
python -m trainer train          # GPU: tens of minutes. CPU: hours.
```

The result lands in `models/`. Verify GPU visibility with:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Using the model in ESPHome

Copy both files somewhere your ESPHome config can reach (or serve them over
HTTP) and reference the manifest:

```yaml
micro_wake_word:
  models:
    - model: models/hey_computer.json
  on_wake_word_detected:
    - voice_assistant.start
```

Requires an ESP32-S3 based device (e.g. the Home Assistant Voice PE, ESP32-S3
Box) and ESPHome ≥ 2024.7.

## Tuning a model that doesn't behave

Wake word training is empirical — the first attempt is rarely perfect. At the
end of `train`, a table of cutoff tradeoffs is printed (false reject rate vs.
false accepts per hour). Work through these in order:

1. **Model won't trigger / triggers falsely at the wrong threshold** — adjust
   `manifest.probability_cutoff` (raise to reduce false accepts, lower to make
   it easier to trigger) and re-run `python -m trainer train --export-only`
   (no retraining needed; the manifest is rewritten).
2. **TTS pronounces the wake word oddly** — add entries to
   `phonetic_spellings` (e.g. `"hey khum_puter"` for *computer*) and
   regenerate samples.
3. **Model is weak in general** — raise `openai.num_samples` (1000+ helps),
   add more real recordings, or raise `training.steps`.
4. **Too many false accepts in daily use** — raise
   `training.negative_class_weight` (e.g. `[25]`) and retrain.
5. **ESPHome logs "failed to allocate tensor arena"** — raise
   `manifest.tensor_arena_size`.

To retrain from scratch after changing sample or feature settings, delete
`training/features` and `training/trained`, then re-run `features` and `train`.

## Project layout

```
config.yaml            all knobs: wake word, TTS settings, training params
trainer/               the pipeline (python -m trainer <command>)
samples/generated/     stage 1 output (TTS clips)
samples/recorded/      put your own recordings here
data/                  downloaded datasets (gitignored)
training/              features, checkpoints, logs (gitignored)
models/                final .tflite + .json (gitignored)
```

## Version pins (why they matter)

`requirements.txt` pins were validated against microWakeWord upstream —
see the comments in that file before upgrading anything:

| Package | Pin | Reason |
|---|---|---|
| tensorflow | 2.18.* | 2.19+/Keras ≥3.11 breaks microWakeWord's metric handling |
| keras | 3.8.0 | same |
| datasets | <4.0 | 4.x requires torchcodec/PyTorch for audio decoding |
| microwakeword | git, editable | non-editable installs drop its `audio`/`layers` subpackages |

## License / data notes

The augmentation and negative datasets have mixed licenses; treat trained
models as suitable for **personal, non-commercial use**. microWakeWord is
Apache-2.0.
