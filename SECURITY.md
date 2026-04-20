# Security Policy

## Supported versions

MLX TTS Studio is currently pre-1.0. Security fixes target the latest code on the default branch.

## Reporting a vulnerability

Please avoid posting exploitable security details in a public issue. Use GitHub's private vulnerability reporting for this repository if it is enabled, or contact the maintainer privately.

Useful reports include:

- A clear description of the issue.
- Steps to reproduce.
- The affected commit or release.
- Any relevant logs with tokens, local paths, or personal data removed.

## Local model and data notes

The app runs locally and writes generated audio to `outputs/`. Model weights are downloaded through Hugging Face into the user's local Hugging Face cache. Do not share local audio, tokens, or model cache paths in public issues unless you have reviewed them first.
