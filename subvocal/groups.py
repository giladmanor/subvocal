"""Training sessions. A session is one folder with two parts: the word test and the sentences.

    data/user-recordings/<speaker>-<YYYY-MM-DD>-<HHMMSS>/
        group.json            the speaker and the start time
        words/                the word test (prompts/probe.tsv): a recording folder
        sentences/            the sentences: a recording folder
        sentence-list.tsv     the sentences that the app selected from the word test
        profile.json          the speaker profile from the word test
        _old/                 earlier recordings of a part that the speaker recorded again

A recording folder has session.json, events.json, full-session.flac, prompts.tsv,
<id>.wav, and hyp.tsv (see subvocal/record.py). The ID of a recording is "<session>/<part>".
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path

from . import review
from .corpus import PROBE_TSV, read_tsv

RECORDINGS = review.RECORDINGS
PARTS = review.PARTS
LIST_FILE = "sentence-list.tsv"


def safe_name(name: str) -> str:
    """Make a speaker name safe for a folder name. Keep Hebrew and Latin letters and digits."""
    name = re.sub(r"[^\w-]+", "-", name.strip(), flags=re.UNICODE).strip("-_")
    return name.lower()


def group_dir(name: str) -> Path:
    folder = (RECORDINGS / name).resolve()
    if folder.parent != RECORDINGS.resolve() or not (folder / "group.json").exists():
        raise ValueError(f"no session: {name}")
    return folder


def create_group(speaker: str) -> str:
    speaker = safe_name(speaker)
    if not speaker:
        raise ValueError("speaker name is empty")
    now = datetime.now()
    folder = RECORDINGS / f"{speaker}-{now.strftime('%Y-%m-%d-%H%M%S')}"
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "group.json").write_text(json.dumps(
        {"speaker": speaker, "created": now.isoformat(timespec="seconds")}, ensure_ascii=False, indent=1))
    return folder.name


def archive(folder: Path, path: Path) -> None:
    """Move a part or a file to _old/. Nothing is deleted."""
    if path.exists():
        old = folder / "_old"
        old.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        shutil.move(str(path), str(old / f"{path.stem}-{stamp}{path.suffix}"))


def part_status(group: Path, part: str) -> dict:
    folder = group / part
    name = f"{group.name}/{part}"
    out = {"name": name, "recorded": (folder / "session.json").exists(), "job": review.JOBS.get(name, {})}
    if not out["recorded"]:
        return out
    meta = json.loads((folder / "session.json").read_text())
    out.update({
        "items": len(meta.get("items", [])), "saved": len(list(folder.glob("*.wav"))),
        "started": meta.get("started", ""), "clock": meta.get("clock"),
        "complete": (folder / "events.json").exists() and (folder / "full-session.flac").exists(),
        "transcribed": (folder / "hyp.tsv").exists() and out["job"].get("state") not in ("live", "running"),
    })
    if out["transcribed"]:
        results = review.load_session(name)["results"].values()
        out["results"] = {
            "total": len(results), "match": sum(r["match"] for r in results),
            "typo": sum(r["same_sound"] for r in results), "extra": sum(r["extra"] for r in results),
        }
        out["results"]["wrong"] = out["results"]["total"] - sum(
            out["results"][k] for k in ("match", "typo", "extra"))
    return out


def list_groups() -> list[dict]:
    out = []
    if not RECORDINGS.exists():
        return out
    for folder in sorted(RECORDINGS.iterdir(), reverse=True):
        if not (folder / "group.json").exists():
            continue
        meta = json.loads((folder / "group.json").read_text())
        out.append({"name": folder.name, "speaker": meta["speaker"], "created": meta["created"],
                    "parts": {p: part_status(folder, p) for p in PARTS}})
    return out


def speakers() -> list[str]:
    return sorted({g["speaker"] for g in list_groups()})


def group_detail(name: str) -> dict:
    folder = group_dir(name)
    meta = json.loads((folder / "group.json").read_text())
    parts = {p: part_status(folder, p) for p in PARTS}
    list_path = folder / LIST_FILE
    return {"name": name, "speaker": meta["speaker"], "created": meta["created"], "parts": parts,
            "list_ready": list_path.exists(),
            "list_count": len(read_tsv(list_path)) if list_path.exists() else 0}


def prepare_part(name: str, part: str) -> tuple[Path, Path, str]:
    """Get the folder and the prompt list for a new recording of a part.

    If the part has a recording, move it to _old/. A new word test also makes the
    sentence list old, because the list comes from the word test.
    """
    if part not in PARTS:
        raise ValueError(f"unknown part: {part}")
    folder = group_dir(name)
    speaker = json.loads((folder / "group.json").read_text())["speaker"]
    if part == "words":
        list_path = PROBE_TSV
        archive(folder, folder / "words")
        archive(folder, folder / LIST_FILE)
        archive(folder, folder / "profile.json")
    else:
        list_path = folder / LIST_FILE
        if not list_path.exists():
            raise ValueError("the sentence list is not ready: record and transcribe the word test first")
        archive(folder, folder / "sentences")
    (folder / part).mkdir()
    return folder / part, list_path, speaker
