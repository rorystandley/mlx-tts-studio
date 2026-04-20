from __future__ import annotations

import contextlib
import gc
import io
import os
import platform
import re
import shutil
import subprocess
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import soundfile as sf
from mlx_audio.tts.generate import generate_audio
from mlx_audio.tts.utils import load_model


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
GENERATION_LOCK = threading.Lock()
LOADED_MODEL: Any | None = None
LOADED_MODEL_ID: str | None = None

CACHE_TABLE_HEADERS = [
    "Model",
    "Good for",
    "Performance on this machine",
    "Status",
    "Size",
    "Cache path",
]


@dataclass(frozen=True)
class ModelPreset:
    key: str
    label: str
    model_id: str
    default_voice: str
    voices: tuple[str, ...]
    lang_code: str
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
    cfg_scale: float
    ddpm_steps: int
    instruct: str
    notes: str
    good_for: str
    min_memory_gb: int
    recommended_memory_gb: int


@dataclass(frozen=True)
class SynthesisResult:
    audio_path: Path
    duration_seconds: float | None
    log: str
    model_id: str
    voice: str | None
    audio_format: str


VOXTRAL_VOICES = (
    "casual_male",
    "casual_female",
    "cheerful_female",
    "neutral_male",
    "neutral_female",
    "fr_male",
    "fr_female",
    "es_male",
    "es_female",
    "de_male",
    "de_female",
    "it_male",
    "it_female",
    "pt_male",
    "pt_female",
    "nl_male",
    "nl_female",
    "hi_male",
    "hi_female",
    "ar_male",
)

KOKORO_VOICES = (
    "af_heart",
    "af_bella",
    "af_nova",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_michael",
    "bf_alice",
    "bf_emma",
    "bm_daniel",
    "bm_george",
)

QWEN_VOICES = ("Chelsie", "Ethan", "Vivian", "Cherry", "Serena", "Dylan")

PRESETS = (
    ModelPreset(
        key="kugel",
        label="KugelAudio 0 Open - best open-source",
        model_id="kugelaudio/kugelaudio-0-open",
        default_voice="default",
        voices=("default", "warm", "clear"),
        lang_code="en",
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        max_tokens=2048,
        cfg_scale=3.0,
        ddpm_steps=16,
        instruct="",
        notes=(
            "MIT licensed 7B model and the best strict open-source preset here. "
            "Very slow first run; 32 GB+ unified memory is recommended."
        ),
        good_for=(
            "Strict open-source/high-quality experiments where license clarity matters "
            "more than speed."
        ),
        min_memory_gb=32,
        recommended_memory_gb=64,
    ),
    ModelPreset(
        key="voxtral",
        label="Voxtral 4B TTS - high quality",
        model_id="mlx-community/Voxtral-4B-TTS-2603-mlx-bf16",
        default_voice="casual_male",
        voices=VOXTRAL_VOICES,
        lang_code="en",
        temperature=0.8,
        top_p=0.95,
        top_k=50,
        max_tokens=4096,
        cfg_scale=1.5,
        ddpm_steps=0,
        instruct="",
        notes=(
            "Open-weight MLX port with strong voice presets; check the model "
            "card license before commercial use."
        ),
        good_for=(
            "Best daily quality target: natural preset voices, multilingual options, "
            "and better quality than small fast models."
        ),
        min_memory_gb=16,
        recommended_memory_gb=32,
    ),
    ModelPreset(
        key="qwen-voice-design",
        label="Qwen3-TTS VoiceDesign - controllable",
        model_id="mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
        default_voice="",
        voices=("",),
        lang_code="english",
        temperature=0.9,
        top_p=1.0,
        top_k=50,
        max_tokens=4096,
        cfg_scale=1.5,
        ddpm_steps=0,
        instruct="A warm, articulate British narrator with natural pacing.",
        notes="Use the instruction field to describe the voice you want.",
        good_for=(
            "Designing a voice from a written description, such as tone, accent, "
            "age, pacing, or presentation style."
        ),
        min_memory_gb=16,
        recommended_memory_gb=24,
    ),
    ModelPreset(
        key="qwen-custom",
        label="Qwen3-TTS CustomVoice - named speakers",
        model_id="mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
        default_voice="Chelsie",
        voices=QWEN_VOICES,
        lang_code="english",
        temperature=0.9,
        top_p=1.0,
        top_k=50,
        max_tokens=4096,
        cfg_scale=1.5,
        ddpm_steps=0,
        instruct="",
        notes=(
            "Named Qwen3 speakers; reference audio can be used by compatible "
            "base models."
        ),
        good_for=(
            "Named speaker presets and controllable Qwen3 voices when you want "
            "repeatable speaker identity."
        ),
        min_memory_gb=16,
        recommended_memory_gb=24,
    ),
    ModelPreset(
        key="chatterbox",
        label="Chatterbox Turbo - expressive cloning",
        model_id="mlx-community/chatterbox-turbo-fp16",
        default_voice="",
        voices=("",),
        lang_code="en",
        temperature=0.8,
        top_p=0.95,
        top_k=1000,
        max_tokens=800,
        cfg_scale=1.5,
        ddpm_steps=0,
        instruct="",
        notes="Fast expressive model; add reference audio for cloning when desired.",
        good_for=(
            "Expressive speech and reference-audio workflows where emotional delivery "
            "or cloning-style output matters."
        ),
        min_memory_gb=12,
        recommended_memory_gb=16,
    ),
    ModelPreset(
        key="kokoro",
        label="Kokoro 82M bf16 - fast local",
        model_id="mlx-community/Kokoro-82M-bf16",
        default_voice="af_heart",
        voices=KOKORO_VOICES,
        lang_code="a",
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        max_tokens=1200,
        cfg_scale=1.5,
        ddpm_steps=0,
        instruct="",
        notes=(
            "Small, Apache-licensed, and excellent for quick drafts on any "
            "Apple Silicon Mac."
        ),
        good_for=(
            "Fast smoke tests, low-latency drafts, and reliable speech on smaller "
            "Apple Silicon machines."
        ),
        min_memory_gb=8,
        recommended_memory_gb=12,
    ),
)

PRESET_BY_LABEL = {preset.label: preset for preset in PRESETS}
PRESET_BY_MODEL_ID = {preset.model_id: preset for preset in PRESETS}
DEFAULT_PRESET_LABEL = "Kokoro 82M bf16 - fast local"
DEFAULT_PRESET = PRESET_BY_LABEL.get(
    os.getenv("DEFAULT_TTS_PRESET_LABEL", DEFAULT_PRESET_LABEL), PRESETS[-1]
)


def get_unified_memory_gb() -> float | None:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            check=True,
            capture_output=True,
            text=True,
        )
        return int(result.stdout.strip()) / 1024**3
    except Exception:
        return None


def clean_log(text: str) -> str:
    return ANSI_RE.sub("", text).strip()


def format_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{num_bytes} B"


def huggingface_hub_cache_dir() -> Path:
    explicit = (
        os.getenv("HF_HUB_CACHE")
        or os.getenv("HUGGINGFACE_HUB_CACHE")
        or os.getenv("TRANSFORMERS_CACHE")
    )
    if explicit:
        return Path(explicit).expanduser()

    hf_home = Path(os.getenv("HF_HOME", "~/.cache/huggingface")).expanduser()
    return hf_home / "hub"


def model_cache_dir(model_id: str) -> Path:
    cache_name = "models--" + model_id.strip().replace("/", "--")
    return huggingface_hub_cache_dir() / cache_name


def model_id_from_cache_dir(path: Path) -> str:
    return path.name.removeprefix("models--").replace("--", "/")


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0

    try:
        result = subprocess.run(
            ["du", "-sk", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
        return int(result.stdout.split()[0]) * 1024
    except Exception:
        total = 0
        for item in path.rglob("*"):
            try:
                if item.is_file() and not item.is_symlink():
                    total += item.stat().st_size
            except OSError:
                continue
        return total


def other_model_description(model_id: str) -> str:
    lowered = model_id.lower()
    if "kokoro" in lowered:
        return "Kokoro-family cached model; usually useful for fast local TTS."
    if "voxtral" in lowered:
        return (
            "Voxtral-family cached model; usually useful for higher-quality "
            "preset-voice TTS."
        )
    if "qwen" in lowered and "tts" in lowered:
        return (
            "Qwen TTS-family cached model; useful for speaker or voice-control "
            "experiments."
        )
    if "chatterbox" in lowered:
        return (
            "Chatterbox-family cached model; useful for expressive or "
            "reference-audio TTS."
        )
    if "kugel" in lowered:
        return (
            "KugelAudio-family cached model; useful for strict open-source "
            "TTS experiments."
        )
    return "Other Hugging Face model cache; not one of this app's built-in TTS presets."


def preset_performance_on_this_machine(preset: ModelPreset) -> str:
    arch = platform.machine()
    memory_gb = get_unified_memory_gb()

    if arch != "arm64":
        return "Not a good fit: MLX acceleration is intended for Apple Silicon arm64."

    if memory_gb is None:
        return (
            "Apple Silicon detected, but memory could not be measured. "
            f"Recommended: {preset.recommended_memory_gb} GB+ unified memory."
        )

    memory_label = f"{memory_gb:.0f} GB"
    if memory_gb < preset.min_memory_gb:
        return (
            f"Poor fit on this machine ({memory_label}): likely to fail, swap, "
            f"or feel very slow. Minimum target: {preset.min_memory_gb} GB."
        )

    if memory_gb < preset.recommended_memory_gb:
        return (
            f"Usable but tight on this machine ({memory_label}): preload first "
            "and avoid running another large local model at the same time. "
            f"Recommended: {preset.recommended_memory_gb} GB+."
        )

    if memory_gb < preset.recommended_memory_gb * 1.5:
        return (
            f"Good fit on this machine ({memory_label}): should run locally, "
            "with first load/download still taking time."
        )

    return (
        f"Comfortable fit on this machine ({memory_label}): should be one of "
        "the smoother choices for local MLX TTS."
    )


def other_model_performance_on_this_machine(model_id: str, cache_size: int) -> str:
    arch = platform.machine()
    memory_gb = get_unified_memory_gb()
    model_id_lower = model_id.lower()

    if arch != "arm64":
        return "Unknown/poor fit: MLX acceleration is intended for Apple Silicon arm64."

    if "35b" in model_id_lower or cache_size >= 15 * 1024**3:
        if memory_gb and memory_gb < 64:
            return (
                f"Likely too heavy for this machine ({memory_gb:.0f} GB) unless "
                "heavily quantized and not used through this TTS app."
            )
        return "Very large model cache; check the model card and runtime before using."

    if "14b" in model_id_lower or cache_size >= 6 * 1024**3:
        if memory_gb and memory_gb < 32:
            return (
                f"Heavy for this machine ({memory_gb:.0f} GB); expect pressure "
                "or failure if another local model is loaded."
            )
        return "Large model cache; should be treated as a heavy runtime."

    if memory_gb:
        return (
            f"Unknown runtime fit on this machine ({memory_gb:.0f} GB): this is "
            "not a built-in TTS preset, so check the model card before loading."
        )

    return "Unknown runtime fit; this is not a built-in TTS preset."


def cache_table_rows(selected_model_id: str = "") -> list[list[str]]:
    selected_model_id = (selected_model_id or "").strip()
    hub_dir = huggingface_hub_cache_dir()

    known_ids = []
    for preset in PRESETS:
        if preset.model_id not in known_ids:
            known_ids.append(preset.model_id)
    if selected_model_id and selected_model_id not in known_ids:
        known_ids.append(selected_model_id)

    rows = []
    seen = set()
    for model_id in known_ids:
        path = model_cache_dir(model_id)
        seen.add(path.name)
        exists = path.exists()
        preset = PRESET_BY_MODEL_ID.get(model_id)
        cache_size = directory_size(path) if exists else 0
        good_for = preset.good_for if preset else other_model_description(model_id)
        performance = (
            preset_performance_on_this_machine(preset)
            if preset
            else other_model_performance_on_this_machine(model_id, cache_size)
        )
        rows.append(
            [
                model_id,
                good_for,
                performance,
                "downloaded" if exists else "not downloaded",
                format_size(cache_size) if exists else "-",
                str(path),
            ]
        )

    if hub_dir.exists():
        for path in sorted(hub_dir.glob("models--*")):
            if path.name in seen or not path.is_dir():
                continue
            model_id = model_id_from_cache_dir(path)
            cache_size = directory_size(path)
            rows.append(
                [
                    model_id,
                    other_model_description(model_id),
                    other_model_performance_on_this_machine(model_id, cache_size),
                    "downloaded (other)",
                    format_size(cache_size),
                    str(path),
                ]
            )

    return rows


def refresh_cache_table(selected_model_id: str = "") -> list[list[str]]:
    return cache_table_rows(selected_model_id)


def delete_selected_model_cache(model_id: str):
    global LOADED_MODEL, LOADED_MODEL_ID

    model_id = (model_id or "").strip()
    if not model_id:
        return "Enter a Hugging Face model id before deleting.", cache_table_rows()

    path = model_cache_dir(model_id)
    hub_dir = huggingface_hub_cache_dir().resolve()
    resolved_path = path.resolve()

    if hub_dir not in resolved_path.parents:
        return (
            f"Refusing to delete outside the Hugging Face hub cache: {path}",
            cache_table_rows(model_id),
        )

    if not path.exists():
        return f"No cache found for {model_id}.", cache_table_rows(model_id)

    with GENERATION_LOCK:
        if LOADED_MODEL_ID == model_id:
            LOADED_MODEL = None
            LOADED_MODEL_ID = None

        size = directory_size(path)
        shutil.rmtree(path)
        mx.clear_cache()
        gc.collect()

    return (
        f"Deleted {model_id} cache and recovered about {format_size(size)}.\n{path}",
        cache_table_rows(model_id),
    )


def preset_to_dict(preset: ModelPreset) -> dict[str, Any]:
    data = asdict(preset)
    data["voices"] = list(preset.voices)
    data["performance_on_this_machine"] = preset_performance_on_this_machine(preset)
    data["cache_path"] = str(model_cache_dir(preset.model_id))
    data["cache_status"] = (
        "downloaded" if model_cache_dir(preset.model_id).exists() else "not downloaded"
    )
    return data


def normalize_model_id(model: str | None) -> str:
    value = (model or "").strip() or DEFAULT_PRESET.model_id
    return PRESET_BY_LABEL[value].model_id if value in PRESET_BY_LABEL else value


def preset_for_model_id(model_id: str) -> ModelPreset | None:
    return PRESET_BY_MODEL_ID.get(model_id)


def safe_prefix(model_id: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_.-]+", "-", model_id).strip("-").lower()
    return f"{int(time.time())}-{stem[:60]}"


def media_type_for_format(audio_format: str) -> str:
    return {"flac": "audio/flac", "wav": "audio/wav"}.get(audio_format, "audio/wav")


def get_loaded_model_id() -> str | None:
    return LOADED_MODEL_ID


def service_health() -> dict[str, Any]:
    memory_gb = get_unified_memory_gb()
    return {
        "status": "ok",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "unified_memory_gb": round(memory_gb, 2) if memory_gb is not None else None,
        "mlx_supported": platform.machine() == "arm64",
        "loaded_model_id": get_loaded_model_id(),
        "default_model": DEFAULT_PRESET.model_id,
        "output_dir": str(OUTPUT_DIR),
        "huggingface_hub_cache_dir": str(huggingface_hub_cache_dir()),
    }


def validate_runtime(model_id: str) -> str:
    model_id = (model_id or "").strip()
    if not model_id:
        raise ValueError("Enter a Hugging Face model id.")
    if platform.machine() != "arm64":
        raise RuntimeError(
            "This app is intended for Apple Silicon Macs running arm64 Python."
        )
    return model_id


def validate_text(text: str | None) -> str:
    text = (text or "").strip()
    if not text:
        raise ValueError("Enter some text to synthesize.")
    return text


def get_model(model_id: str):
    global LOADED_MODEL, LOADED_MODEL_ID

    if LOADED_MODEL is not None and LOADED_MODEL_ID == model_id:
        return LOADED_MODEL

    if LOADED_MODEL is not None:
        del LOADED_MODEL
        LOADED_MODEL = None
        LOADED_MODEL_ID = None
        gc.collect()
        mx.clear_cache()

    LOADED_MODEL = load_model(model_id)
    LOADED_MODEL_ID = model_id
    return LOADED_MODEL


def clear_loaded_model() -> str:
    global LOADED_MODEL, LOADED_MODEL_ID
    with GENERATION_LOCK:
        old_model_id = LOADED_MODEL_ID
        LOADED_MODEL = None
        LOADED_MODEL_ID = None
        gc.collect()
        mx.clear_cache()
    if old_model_id:
        return f"Cleared {old_model_id} from memory."
    return "No model was loaded."


def model_status_note(preset: ModelPreset | None) -> str:
    if not preset:
        return "First run may download model weights from Hugging Face."

    if preset.key == "kugel":
        message = (
            "KugelAudio is a 7B model. First load can take several minutes, "
            "and 32 GB+ unified memory is recommended."
        )
        memory_gb = get_unified_memory_gb()
        if memory_gb and memory_gb < 32:
            message += (
                f" This Mac has {memory_gb:.0f} GB unified memory; "
                "Kokoro is the safest daily driver when another local model is loaded."
            )
        return message

    if preset.key == "voxtral":
        return (
            "Voxtral is a higher-quality preset. Use Download/load first "
            "to pay the initial Hugging Face download cost before generation."
        )

    if preset.key == "kokoro":
        return (
            "Kokoro is the default low-memory preset and the safest choice when "
            "another local LLM is already running."
        )

    return "First run may download model weights from Hugging Face."


def warm_model(model_id: str, preset_label: str | None = None) -> str:
    model_id = validate_runtime(normalize_model_id(model_id))
    preset = PRESET_BY_LABEL.get(preset_label or "") or preset_for_model_id(model_id)
    status_lines = [f"Preparing {model_id}.", model_status_note(preset)]

    with GENERATION_LOCK:
        if LOADED_MODEL is not None and LOADED_MODEL_ID == model_id:
            status_lines.append(f"{model_id} is already loaded in memory.")
            return "\n".join(status_lines)

        load_log = io.StringIO()
        started = time.time()
        try:
            with (
                contextlib.redirect_stdout(load_log),
                contextlib.redirect_stderr(load_log),
            ):
                print(f"Loading model: {model_id}")
                if preset:
                    print(f"Preset: {preset.label}")
                get_model(model_id)
        except Exception:
            load_output = clean_log(load_log.getvalue())
            if load_output:
                status_lines.append(load_output)
            status_lines.append(traceback.format_exc())
            raise RuntimeError(
                f"Model load failed.\n\n{status_lines[-1][-5000:]}"
            ) from None

    load_output = clean_log(load_log.getvalue())
    if load_output:
        status_lines.append(load_output)
    elapsed = time.time() - started
    status_lines.append(f"Ready: {model_id} loaded in {elapsed:.1f}s.")
    return "\n".join(status_lines)


def audio_duration_seconds(audio_path: Path) -> float | None:
    try:
        info = sf.info(str(audio_path))
        return round(info.frames / info.samplerate, 3)
    except Exception:
        return None


def synthesize_text(
    *,
    text: str,
    model_id: str,
    voice: str | None = None,
    lang_code: str | None = None,
    instruct: str | None = None,
    prompt: str | None = None,
    ref_audio: str | None = None,
    ref_text: str | None = None,
    speed: float = 1.0,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    repetition_penalty: float = 1.1,
    cfg_scale: float | None = None,
    ddpm_steps: int | None = None,
    max_tokens: int | None = None,
    audio_format: str = "wav",
    preset_label: str | None = None,
) -> SynthesisResult:
    text = validate_text(text)
    model_id = validate_runtime(normalize_model_id(model_id))
    audio_format = (audio_format or "wav").strip().lower()
    if audio_format not in {"wav", "flac"}:
        raise ValueError("audio_format must be 'wav' or 'flac'.")

    preset = PRESET_BY_LABEL.get(preset_label or "") or preset_for_model_id(model_id)
    output_prefix = safe_prefix(model_id)
    output_path = OUTPUT_DIR / f"{output_prefix}.{audio_format}"
    status_lines = [f"Queued generation with {model_id}."]

    with GENERATION_LOCK:
        cached = LOADED_MODEL is not None and LOADED_MODEL_ID == model_id
        status_lines.append(
            "Using the model already in memory."
            if cached
            else model_status_note(preset)
        )
        load_log = io.StringIO()
        generation_log = io.StringIO()
        try:
            with (
                contextlib.redirect_stdout(load_log),
                contextlib.redirect_stderr(load_log),
            ):
                print(f"Loading model: {model_id}")
                if preset:
                    print(f"Preset: {preset.label}")
                model = get_model(model_id)

            load_output = clean_log(load_log.getvalue())
            if load_output:
                status_lines.append(load_output)

            status_lines.append("Model loaded. Generating audio.")

            with (
                contextlib.redirect_stdout(generation_log),
                contextlib.redirect_stderr(generation_log),
            ):
                print("Generating audio...")
                generate_audio(
                    text=text,
                    model=model,
                    voice=(voice or None),
                    prompt=((prompt or "").strip() or None),
                    instruct=((instruct or "").strip() or None),
                    speed=float(speed),
                    lang_code=((lang_code or "").strip() or "en"),
                    cfg_scale=float(cfg_scale if cfg_scale is not None else 1.5),
                    ddpm_steps=int(ddpm_steps) if ddpm_steps else None,
                    ref_audio=ref_audio or None,
                    ref_text=((ref_text or "").strip() or None),
                    output_path=str(OUTPUT_DIR),
                    file_prefix=output_prefix,
                    audio_format=audio_format,
                    join_audio=True,
                    play=False,
                    verbose=True,
                    temperature=float(temperature if temperature is not None else 0.7),
                    max_tokens=int(max_tokens if max_tokens is not None else 1200),
                    top_p=float(top_p if top_p is not None else 0.9),
                    top_k=int(top_k if top_k is not None else 50),
                    repetition_penalty=float(repetition_penalty),
                )
            generation_output = clean_log(generation_log.getvalue())
            if generation_output:
                status_lines.append(generation_output)
        except Exception:
            load_output = clean_log(load_log.getvalue())
            if load_output and load_output not in status_lines:
                status_lines.append(load_output)
            generation_output = clean_log(generation_log.getvalue())
            if generation_output:
                status_lines.append(generation_output)
            status_lines.append(traceback.format_exc())

    log = "\n".join(line for line in status_lines if line).strip()
    candidates = sorted(OUTPUT_DIR.glob(f"{output_prefix}*.{audio_format}"))
    generated = (
        output_path if output_path.exists() else candidates[-1] if candidates else None
    )
    if generated is None:
        raise RuntimeError(
            f"Generation failed before audio was written.\n\n{log[-5000:]}"
        )

    return SynthesisResult(
        audio_path=generated,
        duration_seconds=audio_duration_seconds(generated),
        log=f"{log}\n\nSaved: {generated}",
        model_id=model_id,
        voice=voice or None,
        audio_format=audio_format,
    )
