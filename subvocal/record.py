"""Record a speaker as they read a list of prompts. Saves 16 kHz mono WAV files.

For the probe list, the screen shows the carrier phrase "המילה היא <word>".
The speaker reads the full phrase. The carrier phrase gives the same start to each item.

Controls: press Enter to start the recording, press Enter again to stop it.
Type "r" and Enter to record the last item again. Type "q" and Enter to stop.

Usage:
    python -m subvocal.record --list prompts/probe.tsv --out data/dana/probe
    python -m subvocal.record --list data/dana/next.tsv --out data/dana/s1
"""

from __future__ import annotations

import argparse
import queue
from pathlib import Path

from .corpus import read_tsv

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", type=Path, required=True, help="probe.tsv or a selection TSV")
    ap.add_argument("--out", type=Path, required=True, help="output folder")
    args = ap.parse_args()

    import soundfile as sf

    rows = read_tsv(args.list)
    args.out.mkdir(parents=True, exist_ok=True)
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
