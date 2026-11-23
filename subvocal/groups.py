"""Training sessions. A session is one folder with two parts: the word test and the sentences.

    data/user-recordings/<speaker>-<YYYY-MM-DD>-<HHMMSS>/
        group.json            the speaker and the start time
        words/                the word test (prompts/probe.tsv): a recording folder
        sentences/            the sentences: a recording folder (the first batch)
        sentences-2/, ...     more batches of sentences, each with its own list
        test/                 the fixed test set (prompts/test-set.tsv): never used for training
        sentence-list.tsv     the sentences that the app selected from the word test
        sentence-list-2.tsv   the list of the second batch, and so on
        profile.json          the speaker profile from the word test
        _old/                 earlier recordings of a part that the speaker recorded again

A recording folder has session.json, events.json, full-session.flac, prompts.tsv,
<id>.wav, and hyp.tsv (see subvocal/record.py). The ID of a recording is "<session>/<part>".
"""

from __future__ import annotations

import json
import math
import re
import shutil
from datetime import datetime
from pathlib import Path

from . import review
from .corpus import PROBE_TSV, read_tsv
from .testset import TEST_TSV

RECORDINGS = review.RECORDINGS
PARTS = review.PARTS
LIST_FILE = "sentence-list.tsv"
CARRIER_TEXT = "המילה היא"  # the carrier phrase of the word test (no niqqud)


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


def batch_part(k: int) -> str:
    return "sentences" if k == 1 else f"sentences-{k}"


def batch_list(k: int) -> str:
    return LIST_FILE if k == 1 else f"sentence-list-{k}.tsv"


def batch_index(part: str) -> int:
    return 1 if part == "sentences" else int(part.split("-")[1])


def batches(folder: Path) -> list[str]:
    """The sentence parts that have a recording, in order: sentences, sentences-2, ..."""
    found = [p.name for p in folder.iterdir() if p.is_dir() and review.PART_RE.match(p.name)
             and p.name.startswith("sentences") and (p / "session.json").exists()]
    return sorted(found, key=batch_index)


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
        parts = {p: part_status(folder, p) for p in PARTS}
        out.append({"name": folder.name, "speaker": meta["speaker"], "created": meta["created"],
                    "parts": parts, "batches": [part_status(folder, b) for b in batches(folder)]})
    return out


def speakers() -> list[str]:
    return sorted({g["speaker"] for g in list_groups()})


def group_detail(name: str) -> dict:
    folder = group_dir(name)
    meta = json.loads((folder / "group.json").read_text())
    parts = {p: part_status(folder, p) for p in PARTS}
    list_path = folder / LIST_FILE
    batch_status = [part_status(folder, b) for b in batches(folder)]
    for b in batch_status:
        b["part"] = b["name"].split("/")[1]
        b["index"] = batch_index(b["part"])
    # The test set is enough one time for each speaker. Find if another session has it.
    test_elsewhere = any(g["speaker"] == meta["speaker"] and g["name"] != name and g["parts"]["test"]["recorded"]
                         for g in list_groups())
    return {"name": name, "speaker": meta["speaker"], "created": meta["created"], "parts": parts,
            "test_elsewhere": test_elsewhere, "batches": batch_status,
            "training_data": training_data(meta["speaker"]),
            "list_ready": list_path.exists(),
            "list_count": len(read_tsv(list_path)) if list_path.exists() else 0}


POOL_TARGETS = (15, 30, 60)  # minutes of sentences. See "Dataset size" in BRIEF.md.


def training_data(speaker: str) -> dict:
    """How much sentence data the speaker has for training, and how much more is necessary."""
    from .corpus import SENTENCES_TSV
    from .testset import test_ids
    from .train import collect

    data = collect(speaker)["train"]  # the sentences of all the sessions, without the test set
    clips = len(data)
    seconds = sum(x["seconds"] for x in data)
    avg = seconds / clips if clips else 4.5  # seconds for each sentence
    recorded = {x["id"] for x in data}
    pool_left = len(read_tsv(SENTENCES_TSV)) - len(test_ids()) - len(recorded)
    targets = []
    for t in POOL_TARGETS:
        need = max(0.0, t * 60 - seconds)
        targets.append({"minutes": t, "reached": need == 0,
                        "sentences": math.ceil(need / avg) if need else 0,
                        "possible": math.ceil(need / avg) <= pool_left})
    return {"clips": clips, "minutes": round(seconds / 60, 2), "avg_seconds": round(avg, 2),
            "pool_left": pool_left, "pool_minutes": round(pool_left * avg / 60, 1), "targets": targets}


def new_batch(name: str, n: int) -> str:
    """Make the list of the next sentence batch of a session, and return its part name.

    The list skips the sentences that the speaker recorded in any session, and the test set.
    """
    from .plan import make_list

    folder = group_dir(name)
    k = len(batches(folder)) + 1
    if k == 1 and (folder / LIST_FILE).exists():
        return batch_part(1)
    if not (folder / batch_list(k)).exists():
        here = [w.stem for b in batches(folder) for w in (folder / b).glob("*.wav")]
        make_list(name, n, out=batch_list(k), exclude=here)
    return batch_part(k)


def prepare_part(name: str, part: str) -> tuple[Path, Path, str]:
    """Get the folder and the prompt list for a new recording of a part.

    If the part has a recording, move it to _old/. A new word test also makes the
    sentence list old, because the list comes from the word test.
    """
    if not review.PART_RE.match(part):
        raise ValueError(f"unknown part: {part}")
    folder = group_dir(name)
    speaker = json.loads((folder / "group.json").read_text())["speaker"]
    if part == "words":
        list_path = PROBE_TSV
        archive(folder, folder / "words")
        for f in folder.glob("sentence-list*.tsv"):  # the lists come from the word test
            archive(folder, f)
        archive(folder, folder / "profile.json")
    elif part == "test":
        list_path = TEST_TSV
        archive(folder, folder / "test")
    else:
        list_path = folder / batch_list(batch_index(part))
        if not list_path.exists():
            raise ValueError("the sentence list is not ready: record and transcribe the word test first")
        archive(folder, folder / part)
    (folder / part).mkdir()
    return folder / part, list_path, speaker
