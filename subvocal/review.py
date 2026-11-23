"""Read recorded sessions for the review screen, and run the transcription of a session.

A session folder has session.json, events.json, full-session.flac, and <id>.wav files.
See subvocal/record.py.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

from .corpus import read_tsv
from .diagnose import normalize, strip_carrier
from .phonetics import REPO_ROOT

RECORDINGS = REPO_ROOT / "data" / "user-recordings"
MIN_TAKE = 0.25  # seconds. A shorter take means the speaker only passed over the card.
TAIL = 0.25      # seconds after each card, because the speaker can finish late.


def compute_takes(events: list[dict], end_frame: int, rate: int) -> list[dict]:
    """All the takes of all the cards, in order. The last take of each card is "kept"."""
    takes = []
    for i, ev in enumerate(events):
        if ev["type"] != "show":
            continue
        start = ev["t"]
        end = events[i + 1]["t"] if i + 1 < len(events) else end_frame
        takes.append({
            "id": ev["id"], "index": ev["index"], "start": start / rate, "end": end / rate,
            "cut_end": min(end + int(TAIL * rate), end_frame) / rate,
            "short": (end - start) < MIN_TAKE * rate, "kept": False,
        })
    last: dict[str, dict] = {}
    for t in takes:
        if not t["short"]:
            last[t["id"]] = t
    for t in last.values():
        t["kept"] = True
    return takes


def pauses(events: list[dict], rate: int) -> list[dict]:
    out = []
    for i, ev in enumerate(events):
        if ev["type"] == "pause" and i + 1 < len(events):
            out.append({"start": ev["t"] / rate, "end": events[i + 1]["t"] / rate})
    return out


def session_dir(name: str) -> Path:
    folder = (RECORDINGS / name).resolve()
    if folder.parent != RECORDINGS.resolve() or not folder.is_dir():
        raise ValueError(f"no session: {name}")
    return folder


def list_sessions() -> list[dict]:
    out = []
    if not RECORDINGS.exists():
        return out
    for folder in sorted(RECORDINGS.iterdir(), reverse=True):
        meta_path = folder / "session.json"
        if not folder.is_dir() or not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        out.append({
            "name": folder.name, "speaker": meta.get("speaker", ""), "started": meta.get("started", ""),
            "list": meta.get("list", ""), "items": len(meta.get("items", [])),
            "saved": len(list(folder.glob("*.wav"))), "transcribed": (folder / "hyp.tsv").exists(),
            "complete": (folder / "events.json").exists() and (folder / "full-session.flac").exists(),
        })
    return out


def is_match(item: dict, hyp: str) -> bool:
    """True if the transcript has the correct word (probe) or the correct sentence."""
    if item.get("carrier"):
        accepted = {item["plain"], *normalize(item.get("accept", "")).split()}
        return bool(accepted & set(strip_carrier(hyp).split()))
    return normalize(hyp) == normalize(item["plain"])


def load_session(name: str) -> dict:
    folder = session_dir(name)
    meta = json.loads((folder / "session.json").read_text())
    events = json.loads((folder / "events.json").read_text())["events"] if (folder / "events.json").exists() else []
    rate = meta.get("rate", 16000)
    end_frame = events[-1]["t"] if events else 0
    items = meta["items"]
    for it in items:  # sessions from before the niqqud change
        it.setdefault("plain", it["text"])
    hyp = {r["id"]: r["text"] for r in read_tsv(folder / "hyp.tsv")} if (folder / "hyp.tsv").exists() else {}
    by_id = {it["id"]: it for it in items}
    results = {i: {"hyp": h, "match": is_match(by_id[i], h)} for i, h in hyp.items() if i in by_id}
    return {
        "name": name, "meta": {k: v for k, v in meta.items() if k != "items"}, "items": items,
        "duration": end_frame / rate, "takes": compute_takes(events, end_frame, rate),
        "pauses": pauses(events, rate), "results": results,
        "transcribe": JOBS.get(name, {}),
    }


# ---------------------------------------------------------------- transcription jobs

JOBS: dict[str, dict] = {}


def start_transcription(name: str) -> dict:
    folder = session_dir(name)
    job = JOBS.get(name)
    if job and job["state"] == "running":
        return job
    total = len(list(folder.glob("*.wav")))
    job = {"state": "running", "done": 0, "total": total, "error": ""}
    JOBS[name] = job

    def run():
        cmd = [sys.executable, "-m", "subvocal.transcribe", "--audio", str(folder), "--out", str(folder / "hyp.tsv")]
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        tail = []
        for line in proc.stdout:
            tail = (tail + [line.rstrip()])[-5:]
            if "\t" in line:
                job["done"] += 1
        proc.wait()
        job["state"] = "done" if proc.returncode == 0 else "error"
        if proc.returncode != 0:
            job["error"] = "\n".join(tail)

    threading.Thread(target=run, daemon=True).start()
    return job
