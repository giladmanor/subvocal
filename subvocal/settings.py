"""The app settings, in data/settings.json. A value that is not in the file gets its default."""

from __future__ import annotations

import copy
import json

from .phonetics import REPO_ROOT

SETTINGS = REPO_ROOT / "data" / "settings.json"
MODELS = REPO_ROOT / "data" / "models"

# The base transcription models. "hf" is for the training, "ct2" is for faster-whisper.
BASES = {
    "turbo": {"label": "ivrit-ai/whisper-large-v3-turbo", "hf": "ivrit-ai/whisper-large-v3-turbo",
              "ct2": "ivrit-ai/whisper-large-v3-turbo-ct2"},
    "large-v3": {"label": "ivrit-ai/whisper-large-v3", "hf": "ivrit-ai/whisper-large-v3",
                 "ct2": "ivrit-ai/whisper-large-v3-ct2"},
}
# Local language models (llama.cpp, GGUF files in data/models/llm/). See subvocal/llm.py.
LOCAL_LLMS = {
    "dictalm-3.0-12b": {"label": "DictaLM 3.0 12B Instruct (מקומי)",
                        "repo": "dicta-il/DictaLM-3.0-Nemotron-12B-Instruct-GGUF",
                        "file": "DictaLM-3.0-Nemotron-12B-Instruct-Q4_K_M.gguf"},
}
DEFAULT_LOCAL_LLM = "dictalm-3.0-12b"
API_LLMS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
LLM_MODELS = [*LOCAL_LLMS, *API_LLMS]

DEFAULTS = {
    "plan_sentences": 40,
    "asr": {"base": "turbo"},
    "train": {"epochs": 5, "lr": 0.0005, "rank": 32, "batch": 4, "with_words": False},
    "active": {},  # speaker -> the name of the training run that the speaker uses ("" = the base model)
    "llm": {"enabled": True, "model": DEFAULT_LOCAL_LLM},
    "tts": {"voice": "Carmit", "rate": 180, "speak": True},
}

LIMITS = {
    ("plan_sentences",): (5, 300), ("train", "epochs"): (1, 50), ("train", "lr"): (1e-6, 1e-2),
    ("train", "rank"): (2, 128), ("train", "batch"): (1, 16), ("tts", "rate"): (80, 400),
}


def _merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict) and k != "active":
            out[k] = _merge(out[k], v)
        elif k in out:
            out[k] = v
    return out


def load() -> dict:
    saved = json.loads(SETTINGS.read_text()) if SETTINGS.exists() else {}
    return _merge(DEFAULTS, saved)


def save(changes: dict) -> dict:
    """Merge the changes into the settings, check them, and write the file."""
    values = _merge(load(), changes)
    for path, (lo, hi) in LIMITS.items():
        v = values
        for k in path:
            v = v[k]
        if not lo <= float(v) <= hi:
            raise ValueError(f"{'.'.join(path)} must be from {lo} to {hi}")
    if values["asr"]["base"] not in BASES:
        raise ValueError(f"unknown base model: {values['asr']['base']}")
    if values["llm"]["model"] not in LLM_MODELS:
        raise ValueError(f"unknown language model: {values['llm']['model']}")
    SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS.write_text(json.dumps(values, ensure_ascii=False, indent=1))
    return values


def base() -> dict:
    return BASES[load()["asr"]["base"]]
