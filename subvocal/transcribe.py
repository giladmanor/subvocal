"""Transcribe a folder of WAV files with a Hebrew Whisper model (faster-whisper, CTranslate2).

Each file name (without ".wav") is the item ID, for example P01.wav or S042.wav.

Usage:
    python -m subvocal.transcribe --audio data/dana/probe --out data/dana/probe/hyp.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .corpus import write_tsv

DEFAULT_MODEL = "ivrit-ai/whisper-large-v3-turbo-ct2"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", type=Path, required=True, help="folder with <id>.wav files")
    ap.add_argument("--out", type=Path, required=True, help="output TSV (id, text)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--compute-type", default="int8", help="int8 for CPU (Apple Silicon)")
    args = ap.parse_args()

    import numpy as np
    import soundfile as sf
    from faster_whisper import WhisperModel

    model = WhisperModel(args.model, device="cpu", compute_type=args.compute_type)
    rows = []
    for wav in sorted(args.audio.glob("*.wav")):
        audio, rate = sf.read(wav, dtype="float32", always_2d=True)
        if rate != 16000:
            raise ValueError(f"{wav}: sample rate is {rate}, expected 16000")
        # Send samples, not a path. This skips the PyAV decoder.
        segments, _ = model.transcribe(
            np.ascontiguousarray(audio.mean(axis=1)), language="he", beam_size=5,
            condition_on_previous_text=False, without_timestamps=True,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        rows.append({"id": wav.stem, "text": text})
        print(f"{wav.stem}\t{text}")
    write_tsv(args.out, rows, ["id", "text"])
    print(f"Wrote {len(rows)} transcripts to {args.out}")


if __name__ == "__main__":
    main()
