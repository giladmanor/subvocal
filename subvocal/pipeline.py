"""The status of the three stages of the pipeline, and the jobs that the pipeline screen starts.

1. Transcription: the base Whisper model, or a fine-tuned model of the speaker (subvocal/train.py).
2. Context correction with a language model: not in the first version. The screen keeps its settings.
3. Speech: the macOS "say" command, with a voice and a rate.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
import threading

from . import review, settings
from .groups import list_groups
from .phonetics import REPO_ROOT
from .trial import llm_credentials


def runs(speaker: str) -> list[dict]:
    out = []
    folder = settings.MODELS / speaker
    for run in sorted(folder.iterdir(), reverse=True) if folder.exists() else []:
        f = run / "run.json"
        if not f.exists():
            continue
        info = json.loads(f.read_text())
        ev = info.get("eval", {})
        out.append({
            "name": run.name, "created": info.get("created"), "state": info.get("state"),
            "ready": (run / "ct2" / "model.bin").exists(), "base": info.get("base"),
            "params": info.get("params", {}), "data": info.get("data", {}),
            "history": info.get("history", []), "best_epoch": info.get("best_epoch"),
            "eval": {split: {k: {m: r[k][m] for m in ("wer", "cer", "n")} for k in ("base", "tuned")}
                     for split, r in ev.items()},
        })
    return out


def voices() -> list[dict]:
    """The voices of "say" for Hebrew (he_IL)."""
    out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
    found = []
    for line in out.splitlines():
        m = re.match(r"^(.+?)\s+([a-z]{2}_[A-Z]{2})\s+#", line)
        if m and m.group(2) == "he_IL":
            found.append({"name": m.group(1).strip(), "locale": m.group(2)})
    return found


def status(speaker: str | None) -> dict:
    from .train import summary

    cfg = settings.load()
    speakers = sorted({g["speaker"] for g in list_groups()})
    speaker = speaker if speaker in speakers else (speakers[0] if speakers else "")
    active = cfg["active"].get(speaker, "")
    all_runs = runs(speaker) if speaker else []
    return {
        "speaker": speaker, "speakers": speakers, "settings": cfg,
        "bases": settings.BASES, "llm_models": settings.LLM_MODELS,
        "asr": {"active": active if any(r["name"] == active and r["ready"] for r in all_runs) else "",
                "runs": all_runs,
                "data": summary(speaker, cfg["train"]["with_words"]) if speaker else {},
                "job": review.JOBS.get(f"train:{speaker}", {})},
        "tts": {"engine": "macOS say", "os": f"macOS {platform.mac_ver()[0]}", "voices": voices()},
        "llm_credentials": llm_credentials(),
    }


def start_training(speaker: str) -> dict:
    """Run subvocal/train.py in a subprocess, and read its progress lines."""
    name = f"train:{speaker}"
    job = review.JOBS.get(name)
    if job and job["state"] == "running":
        return job
    job = {"state": "running", "stage": "start", "done": 0, "total": 0, "error": "", "message": ""}
    review.JOBS[name] = job

    def run():
        cmd = [sys.executable, "-u", "-m", "subvocal.train", "--speaker", speaker]
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        tail: list[str] = []
        for line in proc.stdout:
            if line.startswith("@@"):
                try:
                    job.update(json.loads(line[2:]))
                except json.JSONDecodeError:
                    pass
            else:
                tail = (tail + [line.rstrip()])[-8:]
        proc.wait()
        job["state"] = "done" if proc.returncode == 0 else "error"
        if proc.returncode != 0:
            job["error"] = "\n".join(tail)

    threading.Thread(target=run, daemon=True).start()
    return job


def set_active(speaker: str, run: str) -> dict:
    if run and not any(r["name"] == run and r["ready"] for r in runs(speaker)):
        raise ValueError(f"no ready model: {run}")
    active = {**settings.load()["active"], speaker: run}
    return settings.save({"active": active})


def say_sample(text: str) -> dict:
    cfg = settings.load()["tts"]
    subprocess.Popen(["say", "-v", cfg["voice"], "-r", str(int(cfg["rate"])), text])
    return {"ok": True}
