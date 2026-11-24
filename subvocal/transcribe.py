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
# The app transcribes while the speaker records. Keep some CPU cores free for the recording.
APP_THREADS = 6


def load_model(name: str = DEFAULT_MODEL, compute_type: str = "int8", cpu_threads: int = 0):
    from faster_whisper import WhisperModel

    return WhisperModel(name, device="cpu", compute_type=compute_type, cpu_threads=cpu_threads)


def transcribe_audio(model, audio) -> str:
    """Transcribe 16 kHz mono float32 samples. Send samples, not a path: this skips the PyAV decoder."""
    import numpy as np

    segments, _ = model.transcribe(
        np.ascontiguousarray(audio, dtype="float32"), language="he", beam_size=5,
        condition_on_previous_text=False, without_timestamps=True,
    )
    return " ".join(s.text.strip() for s in segments).strip()


def transcribe_words(model, audio) -> list[dict]:
    """Transcribe, and give each word its start and end time (seconds) and its probability."""
    import numpy as np

    segments, _ = model.transcribe(
        np.ascontiguousarray(audio, dtype="float32"), language="he", beam_size=5,
        condition_on_previous_text=False, word_timestamps=True,
    )
    # "join": the piece has no space before it, so it is the end of the word before it
    # (Whisper cuts צ'ילי into "צ" and "'ילי").
    return [{"word": w.word.strip(), "join": not w.word.startswith(" ") and i > 0,
             "start": w.start, "end": w.end, "prob": round(w.probability, 3)}
            for i, w in enumerate(w for s in segments for w in (s.words or [])) if w.word.strip()]


def read_wav(path: Path):
    import soundfile as sf

    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != 16000:
        raise ValueError(f"{path}: sample rate is {rate}, expected 16000")
    return audio.mean(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", type=Path, required=True, help="folder with <id>.wav files")
    ap.add_argument("--out", type=Path, required=True, help="output TSV (id, text)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--compute-type", default="int8", help="int8 for CPU (Apple Silicon)")
    args = ap.parse_args()

    model = load_model(args.model, args.compute_type)
    rows = []
    for wav in sorted(args.audio.glob("*.wav")):
        text = transcribe_audio(model, read_wav(wav))
        rows.append({"id": wav.stem, "text": text})
        print(f"{wav.stem}\t{text}", flush=True)
    write_tsv(args.out, rows, ["id", "text"])
    print(f"Wrote {len(rows)} transcripts to {args.out}")


if __name__ == "__main__":
    main()
