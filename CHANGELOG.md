# Changelog

All notable changes to MLX TTS Studio will be documented here.

## 0.1.0 - 2026-04-20

- Added a Gradio UI for Apple Silicon text-to-speech with MLX Audio.
- Added built-in presets for Kokoro, Voxtral, Qwen3 TTS, Chatterbox, and KugelAudio.
- Added model download/warmup controls.
- Added a model cache viewer with cache sizes, model descriptions, and machine-specific performance notes.
- Added selected model cache deletion.
- Added a FastAPI service with `/health`, `/models`, `/synthesize`, and `/v1/audio/speech`.
- Defaulted to Kokoro for a lower-memory startup path.
