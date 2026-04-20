from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any

import gradio as gr
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from tts_service import (
    CACHE_TABLE_HEADERS,
    DEFAULT_PRESET,
    PRESETS,
    cache_table_rows,
    clear_loaded_model,
    delete_selected_model_cache,
    get_unified_memory_gb,
    media_type_for_format,
    normalize_model_id,
    preset_for_model_id,
    preset_to_dict,
    refresh_cache_table,
    service_health,
    synthesize_text,
    warm_model,
)


class SynthesisRequest(BaseModel):
    text: str | None = None
    input: str | None = None
    model: str | None = None
    voice: str | None = None
    lang_code: str | None = None
    instruct: str | None = None
    prompt: str | None = None
    ref_audio: str | None = None
    ref_text: str | None = None
    speed: float = 1.0
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float = 1.1
    cfg_scale: float | None = None
    ddpm_steps: int | None = None
    max_tokens: int | None = None
    audio_format: str = "wav"


class OpenAISpeechRequest(BaseModel):
    input: str = Field(..., min_length=1)
    model: str | None = None
    voice: str | None = None
    response_format: str = "wav"
    speed: float = 1.0
    instructions: str | None = None


api_app = FastAPI(
    title="MLX TTS Studio",
    description="Local Apple Silicon TTS API backed by MLX Audio models.",
    version="0.1.0",
)

APP_ICON_PATH = Path(__file__).parent / "docs" / "assets" / "mlx-tts-studio-icon.svg"
APP_ICON_URL = "/favicon.ico?v=mlx-tts-studio-icon-1"
APP_HEAD = f"""
<link rel="icon" type="image/svg+xml" href="{APP_ICON_URL}" />
<link rel="shortcut icon" type="image/svg+xml" href="{APP_ICON_URL}" />
"""


def request_text(body: SynthesisRequest) -> str:
    return (body.text or body.input or "").strip()


def service_kwargs_from_request(body: SynthesisRequest) -> dict[str, Any]:
    model_id = normalize_model_id(body.model)
    preset = preset_for_model_id(model_id) or DEFAULT_PRESET
    return {
        "text": request_text(body),
        "model_id": model_id,
        "voice": body.voice if body.voice is not None else preset.default_voice,
        "lang_code": body.lang_code or preset.lang_code,
        "instruct": body.instruct if body.instruct is not None else preset.instruct,
        "prompt": body.prompt,
        "ref_audio": body.ref_audio,
        "ref_text": body.ref_text,
        "speed": body.speed,
        "temperature": (
            body.temperature if body.temperature is not None else preset.temperature
        ),
        "top_p": body.top_p if body.top_p is not None else preset.top_p,
        "top_k": body.top_k if body.top_k is not None else preset.top_k,
        "repetition_penalty": body.repetition_penalty,
        "cfg_scale": body.cfg_scale if body.cfg_scale is not None else preset.cfg_scale,
        "ddpm_steps": body.ddpm_steps
        if body.ddpm_steps is not None
        else preset.ddpm_steps,
        "max_tokens": body.max_tokens
        if body.max_tokens is not None
        else preset.max_tokens,
        "audio_format": body.audio_format,
        "preset_label": preset.label,
    }


def synthesize_or_raise(**kwargs):
    try:
        return synthesize_text(**kwargs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@api_app.get("/health")
def health():
    return service_health()


@api_app.get("/models")
def models(selected_model_id: str = ""):
    return {
        "default_model": DEFAULT_PRESET.model_id,
        "loaded_model_id": service_health()["loaded_model_id"],
        "presets": [preset_to_dict(preset) for preset in PRESETS],
        "cache_headers": CACHE_TABLE_HEADERS,
        "cache": cache_table_rows(selected_model_id),
    }


@api_app.post("/synthesize")
def synthesize_api(body: SynthesisRequest):
    result = synthesize_or_raise(**service_kwargs_from_request(body))
    return {
        "audio_path": str(result.audio_path),
        "duration_seconds": result.duration_seconds,
        "model": result.model_id,
        "voice": result.voice,
        "format": result.audio_format,
        "log": result.log,
    }


@api_app.post("/v1/audio/speech")
def openai_speech_api(body: OpenAISpeechRequest):
    response_format = (body.response_format or "wav").lower()
    if response_format not in {"wav", "flac"}:
        raise HTTPException(
            status_code=400,
            detail="response_format must be 'wav' or 'flac' for this local service.",
        )

    model_id = normalize_model_id(body.model)
    preset = preset_for_model_id(model_id) or DEFAULT_PRESET
    result = synthesize_or_raise(
        text=body.input,
        model_id=model_id,
        voice=body.voice if body.voice is not None else preset.default_voice,
        lang_code=preset.lang_code,
        instruct=body.instructions
        if body.instructions is not None
        else preset.instruct,
        speed=body.speed,
        temperature=preset.temperature,
        top_p=preset.top_p,
        top_k=preset.top_k,
        cfg_scale=preset.cfg_scale,
        ddpm_steps=preset.ddpm_steps,
        max_tokens=preset.max_tokens,
        audio_format=response_format,
        preset_label=preset.label,
    )
    return FileResponse(
        str(result.audio_path),
        media_type=media_type_for_format(response_format),
        filename=result.audio_path.name,
    )


def status_text(lines: list[str], line: str) -> str:
    lines.append(line)
    return "\n".join(lines)


def preset_changed(label: str):
    preset = next(item for item in PRESETS if item.label == label)
    ddpm_value = preset.ddpm_steps if preset.ddpm_steps else None
    return (
        preset.model_id,
        gr.update(choices=list(preset.voices), value=preset.default_voice),
        preset.lang_code,
        preset.temperature,
        preset.top_p,
        preset.top_k,
        preset.max_tokens,
        preset.cfg_scale,
        ddpm_value,
        preset.instruct,
        preset.notes,
    )


def download_model_ui(preset_label: str, model_id: str):
    model_id = (model_id or "").strip()
    if not model_id:
        yield "Enter a Hugging Face model id."
        return

    yield f"Preparing {model_id}."
    try:
        yield warm_model(model_id, preset_label)
    except Exception as exc:
        yield str(exc)


def generate_speech(
    text: str,
    preset_label: str,
    model_id: str,
    voice: str,
    lang_code: str,
    instruct: str,
    prompt: str,
    ref_audio: str | None,
    ref_text: str,
    speed: float,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    cfg_scale: float,
    ddpm_steps: int | None,
    max_tokens: int,
    audio_format: str,
):
    yield None, None, f"Queued generation with {(model_id or '').strip()}."
    try:
        result = synthesize_text(
            text=text,
            model_id=model_id,
            voice=voice,
            lang_code=lang_code,
            instruct=instruct,
            prompt=prompt,
            ref_audio=ref_audio,
            ref_text=ref_text,
            speed=speed,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            cfg_scale=cfg_scale,
            ddpm_steps=ddpm_steps,
            max_tokens=max_tokens,
            audio_format=audio_format,
            preset_label=preset_label,
        )
    except Exception as exc:
        yield None, None, str(exc)
        return

    audio_path = str(result.audio_path)
    yield audio_path, audio_path, result.log


def build_app() -> gr.Blocks:
    memory_gb = get_unified_memory_gb()
    memory_label = (
        f"{memory_gb:.0f} GB unified memory" if memory_gb else "memory unknown"
    )
    machine_label = f"{platform.machine()} / {memory_label}"

    with gr.Blocks(title="MLX TTS Studio", fill_height=True) as demo:
        gr.HTML(
            f"""
            <header style="display:flex;align-items:center;gap:14px;margin:0 0 18px;">
              <img
                src="{APP_ICON_URL}"
                alt=""
                aria-hidden="true"
                style="width:52px;height:52px;border-radius:12px;flex:0 0 auto;"
              />
              <div>
                <h1 style="margin:0 0 4px;font-size:28px;line-height:1.1;">MLX TTS Studio</h1>
                <p style="margin:0;color:var(--body-text-color-subdued);">
                  Apple Silicon MLX text-to-speech. <code>{machine_label}</code>
                </p>
              </div>
            </header>
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(
                    label="Text",
                    value=(
                        "Good evening. This voice was generated locally on Apple Silicon "
                        "with MLX and an open Hugging Face model."
                    ),
                    lines=9,
                    max_lines=18,
                    autofocus=True,
                )
                generate_button = gr.Button(
                    "Generate speech", variant="primary", size="lg"
                )
                audio_output = gr.Audio(
                    label="Audio", type="filepath", interactive=False
                )
                file_output = gr.File(label="File")

            with gr.Column(scale=2):
                preset = gr.Dropdown(
                    label="Model preset",
                    choices=[item.label for item in PRESETS],
                    value=DEFAULT_PRESET.label,
                )
                model_id = gr.Textbox(
                    label="Hugging Face model id", value=DEFAULT_PRESET.model_id
                )
                voice = gr.Dropdown(
                    label="Voice",
                    choices=list(DEFAULT_PRESET.voices),
                    value=DEFAULT_PRESET.default_voice,
                    allow_custom_value=True,
                )
                lang_code = gr.Dropdown(
                    label="Language",
                    choices=[
                        "en",
                        "a",
                        "auto",
                        "english",
                        "chinese",
                        "fr",
                        "es",
                        "de",
                        "it",
                        "pt",
                        "nl",
                        "hi",
                        "ar",
                    ],
                    value=DEFAULT_PRESET.lang_code,
                    allow_custom_value=True,
                )
                preset_notes = gr.Textbox(
                    label="Preset notes",
                    value=DEFAULT_PRESET.notes,
                    lines=2,
                    interactive=False,
                )
                download_button = gr.Button("Download/load selected model")
                clear_button = gr.Button("Clear loaded model")

        with gr.Accordion("Voice and sampling controls", open=False):
            with gr.Row():
                instruct = gr.Textbox(
                    label="Instruction",
                    value=DEFAULT_PRESET.instruct,
                    lines=3,
                    placeholder="VoiceDesign prompt, emotion, style, or voice description.",
                )
                prompt = gr.Textbox(
                    label="Prompt",
                    lines=3,
                    placeholder="Optional model-specific prompt prefix.",
                )
            with gr.Row():
                ref_audio = gr.Audio(label="Reference audio", type="filepath")
                ref_text = gr.Textbox(
                    label="Reference text",
                    lines=4,
                    placeholder="Optional transcript for reference audio.",
                )
            with gr.Row():
                speed = gr.Slider(0.5, 1.8, value=1.0, step=0.05, label="Speed")
                temperature = gr.Slider(
                    0.1,
                    1.5,
                    value=DEFAULT_PRESET.temperature,
                    step=0.05,
                    label="Temperature",
                )
                top_p = gr.Slider(
                    0.1, 1.0, value=DEFAULT_PRESET.top_p, step=0.01, label="Top-p"
                )
                top_k = gr.Number(
                    value=DEFAULT_PRESET.top_k, precision=0, label="Top-k"
                )
            with gr.Row():
                repetition_penalty = gr.Slider(
                    1.0,
                    2.0,
                    value=1.1,
                    step=0.01,
                    label="Repetition penalty",
                )
                cfg_scale = gr.Slider(
                    0.0,
                    6.0,
                    value=DEFAULT_PRESET.cfg_scale,
                    step=0.1,
                    label="CFG scale",
                )
                ddpm_steps = gr.Number(
                    value=DEFAULT_PRESET.ddpm_steps or None,
                    precision=0,
                    label="DDPM steps",
                )
                max_tokens = gr.Slider(
                    128,
                    8192,
                    value=DEFAULT_PRESET.max_tokens,
                    step=64,
                    label="Max tokens",
                )
            audio_format = gr.Radio(["wav", "flac"], value="wav", label="Audio format")

        with gr.Accordion("Model cache", open=False):
            cache_table = gr.Dataframe(
                headers=CACHE_TABLE_HEADERS,
                value=cache_table_rows(DEFAULT_PRESET.model_id),
                datatype=["str", "str", "str", "str", "str", "str"],
                interactive=False,
                label="Hugging Face model cache",
            )
            with gr.Row():
                refresh_cache_button = gr.Button("Refresh cache")
                delete_cache_button = gr.Button(
                    "Delete selected model cache", variant="stop"
                )

        logs = gr.Textbox(label="Run log", lines=12, max_lines=24)

        preset.change(
            preset_changed,
            inputs=[preset],
            outputs=[
                model_id,
                voice,
                lang_code,
                temperature,
                top_p,
                top_k,
                max_tokens,
                cfg_scale,
                ddpm_steps,
                instruct,
                preset_notes,
            ],
        )
        download_button.click(
            download_model_ui, inputs=[preset, model_id], outputs=[logs]
        )
        clear_button.click(clear_loaded_model, outputs=[logs])
        refresh_cache_button.click(
            refresh_cache_table, inputs=[model_id], outputs=[cache_table]
        )
        delete_cache_button.click(
            delete_selected_model_cache, inputs=[model_id], outputs=[logs, cache_table]
        )
        generate_button.click(
            generate_speech,
            inputs=[
                text,
                preset,
                model_id,
                voice,
                lang_code,
                instruct,
                prompt,
                ref_audio,
                ref_text,
                speed,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                cfg_scale,
                ddpm_steps,
                max_tokens,
                audio_format,
            ],
            outputs=[audio_output, file_output, logs],
        )

    return demo


def create_app() -> FastAPI:
    demo = build_app()
    demo.queue(default_concurrency_limit=1)
    return gr.mount_gradio_app(
        api_app,
        demo,
        path="/",
        favicon_path=str(APP_ICON_PATH),
        head=APP_HEAD,
    )


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    )
