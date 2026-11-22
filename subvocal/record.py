"""Record a speaker as they read a list of prompts. Saves 16 kHz mono WAV files.

For the probe list, the screen shows the carrier phrase "המילה היא <word>".
The speaker reads the full phrase. The carrier phrase gives the same start to each item.

Controls: press Enter to start the recording, press Enter again to stop it.
Type "r" and Enter to record the last item again. Type "q" and Enter to stop.

Each run asks for the name of the speaker. It then makes a new session folder:

    data/user-recordings/<speaker>-<YYYY-MM-DD>-<HHMMSS>/

The folder gets one WAV file for each item and a copy of the prompt list (prompts.tsv).
The default prompt list is prompts/probe.tsv.

Usage:
    python -m subvocal.record                                   # probe list, asks for the name
    python -m subvocal.record --speaker dana --list data/user-recordings/dana-next.tsv
    python -m subvocal.record --out data/user-recordings/dana-2026-11-22-213000   # continue a session
"""

from __future__ import annotations

import argparse
import queue
import re
import shutil
from datetime import datetime
from pathlib import Path

from .corpus import PROBE_TSV, read_tsv
from .phonetics import REPO_ROOT

RECORDINGS = REPO_ROOT / "data" / "user-recordings"

RATE = 16000


def record_until_enter():
    import numpy as np
    import sounddevice as sd

    q: queue.Queue = queue.Queue()
    with sd.InputStream(samplerate=RATE, channels=1, dtype="int16", callback=lambda d, *_: q.put(d.copy())):
        input("  recording... press Enter to stop")
    chunks = []
    while not q.empty():
        chunks.append(q.get())
    return np.concatenate(chunks) if chunks else np.zeros((0, 1), dtype="int16")


def safe_name(name: str) -> str:
    """Make a speaker name safe for a folder name. Keep Hebrew and Latin letters and digits."""
    name = re.sub(r"[^\w-]+", "-", name.strip(), flags=re.UNICODE).strip("-_")
    return name.lower()


def new_session_dir(speaker: str | None) -> Path:
    while not speaker:
        speaker = safe_name(input("Speaker name: "))
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    return RECORDINGS / f"{safe_name(speaker)}-{stamp}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", type=Path, default=PROBE_TSV, help="prompt list (default: prompts/probe.tsv)")
    ap.add_argument("--speaker", help="speaker name (the script asks if you do not give it)")
    ap.add_argument("--out", type=Path, help="an existing session folder, to continue a session")
    args = ap.parse_args()

    import soundfile as sf

    rows = read_tsv(args.list)
    if args.out is None:
        args.out = new_session_dir(args.speaker)
    args.out.mkdir(parents=True, exist_ok=True)
    if not (args.out / "prompts.tsv").exists():
        shutil.copyfile(args.list, args.out / "prompts.tsv")
    print(f"Prompt list: {args.list} ({len(rows)} items)")
    print(f"Session folder: {args.out}")
    i = 0
    while i < len(rows):
        row = rows[i]
        text = f"המילה היא {row['word']}" if "word" in row else row["text"]
        path = args.out / f"{row['id']}.wav"
        if path.exists():
            i += 1
            continue
        print(f"\n[{i + 1}/{len(rows)}] {row['id']}:  {text}")
        cmd = input("  Enter = start, q = quit: ").strip().lower()
        if cmd == "q":
            break
        sf.write(path, record_until_enter(), RATE)
        if input("  saved. Enter = next, r = again: ").strip().lower() == "r":
            path.unlink()
            continue
        i += 1
    done = sorted(p.stem for p in args.out.glob("*.wav"))
    print(f"\n{len(done)} recordings in {args.out}")


if __name__ == "__main__":
    main()
