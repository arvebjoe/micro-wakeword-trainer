"""Local training pipeline for microWakeWord models (ESPHome / Home Assistant).

Stage 1 (`generate`, `ingest`): collect 16 kHz mono WAV samples of the wake
word — synthesized with OpenAI TTS and/or recorded by the user.

Stage 2 (`download`, `features`, `train`): augment the samples, compute
spectrogram features, train a streaming MixedNet with microWakeWord and export
a quantized `.tflite` model plus the ESPHome v2 JSON manifest.
"""

__version__ = "0.1.0"
