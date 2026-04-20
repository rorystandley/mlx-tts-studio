# Contributing

Thanks for taking a look at MLX TTS Studio. This project is intended to stay small, local-first, and easy to run on Apple Silicon.

## Development setup

```bash
git clone git@github.com:rorystandley/mlx-tts-studio.git
cd mlx-tts-studio
uv sync
./run.sh
```

The app expects arm64 Python on Apple Silicon. Some models are large, so use the Kokoro preset for quick smoke tests.

## Checks

Run these before opening a pull request:

```bash
uv run ruff format .
uv run ruff check .
uv run python -m py_compile app.py tts_service.py
```

If you change API behavior, also verify:

```bash
curl -s http://127.0.0.1:7860/health
curl -s http://127.0.0.1:7860/models
```

## Repository hygiene

Please do not commit:

- Hugging Face model weights or cache directories.
- Generated audio files from `outputs/`.
- Local `.env` files or tokens.
- Virtual environments, editor folders, or OS metadata.

## Model changes

When adding or changing a model preset, include:

- What the model is good for.
- Its expected memory profile.
- A note to check the upstream model card for license and usage terms.
