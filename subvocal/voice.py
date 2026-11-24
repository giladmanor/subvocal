"""Keep only the voice of the speaker: remove the speech of other people before the transcription.

1. Enrollment: the app makes a voice print of the speaker from the training sentences
   (never from the test set), from the sessions that the user selected. A voice print is the average of the speaker embeddings of the clips.
2. Filter: the VAD (Silero, from faster-whisper) finds the speech in a turn. The app cuts the
   speech into short windows and compares each window with the voice print (cosine similarity).
   It removes the parts that are below the threshold, and transcribes only the rest.

The filter works when another person talks before, after, or between the words of the speaker.
It cannot separate two voices that talk at the same time.

The speaker model is WeSpeaker ECAPA-TDNN 1024 (VoxCeleb), ONNX, license Apache-2.0:
https://huggingface.co/Wespeaker/wespeaker-voxceleb-ecapa-tdnn1024-LM
It was the best of four WeSpeaker models on the recordings of this project (see voice_test.py).

    python -m subvocal.voice --enroll gilad      # make the voice print
    python -m subvocal.voice --test gilad        # the noise test (subvocal/voice_test.py)
"""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path

import numpy as np

from . import settings

RATE = 16000
SPEAKER_DIR = settings.MODELS / "speaker"
SPEAKER_REPO = "Wespeaker/wespeaker-voxceleb-ecapa-tdnn1024-LM"
SPEAKER_FILE = "voxceleb_ECAPA1024_LM.onnx"

WINDOW = 1.5      # seconds of audio in one embedding. Shorter windows give less reliable scores.
HOP = 0.25        # seconds between the starts of two windows (also the size of a decision step)
MIN_KEEP = 0.3    # remove a kept part that is shorter than this (seconds)
PAD = 0.15        # add this much audio on each side of a kept part (seconds)
GAP = 0.25        # silence between two kept parts (seconds)
THRESHOLD_SHARE = 0.55  # the threshold is this share of the median score of the speaker's own validation clips
MIN_SPEECH = 0.3   # a window gets a score only if the VAD finds speech in this share of it

_lock = threading.Lock()
_session = None


# ---------------------------------------------------------------- embeddings

def _model():
    global _session
    with _lock:
        if _session is None:
            import onnxruntime as ort

            path = SPEAKER_DIR / SPEAKER_FILE
            if not path.exists():
                from huggingface_hub import hf_hub_download

                SPEAKER_DIR.mkdir(parents=True, exist_ok=True)
                hf_hub_download(SPEAKER_REPO, SPEAKER_FILE, local_dir=SPEAKER_DIR)
            _session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return _session


def _fbank(audio: np.ndarray) -> np.ndarray:
    """80 Kaldi filter-bank features with the mean removed, as WeSpeaker uses them."""
    import kaldi_native_fbank as knf

    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = RATE
    opts.frame_opts.frame_length_ms = 25
    opts.frame_opts.frame_shift_ms = 10
    opts.frame_opts.dither = 0
    opts.mel_opts.num_bins = 80
    opts.energy_floor = 0.0
    fb = knf.OnlineFbank(opts)
    fb.accept_waveform(RATE, (audio * 32768).tolist())
    fb.input_finished()
    feats = np.array([fb.get_frame(i) for i in range(fb.num_frames_ready)], dtype=np.float32)
    return feats - feats.mean(axis=0, keepdims=True)


def embed(audio: np.ndarray) -> np.ndarray:
    """A unit-length speaker embedding (256 values) of 16 kHz float audio."""
    feats = _fbank(audio)
    out = _model().run(None, {"feats": feats[None]})[0][0]
    return out / (np.linalg.norm(out) + 1e-9)


# ---------------------------------------------------------------- voice print

def print_path(speaker: str) -> Path:
    return settings.MODELS / speaker / "voiceprint.json"


def status(speaker: str) -> dict:
    """The voice print and the last noise test of the speaker, for the screen."""
    p, t = print_path(speaker), settings.MODELS / speaker / "voice-test.json"
    out = {"print": None, "test": None}
    if p.exists():
        d = json.loads(p.read_text())
        out["print"] = {k: d.get(k) for k in ("created", "clips", "threshold", "model", "sessions")}
    if t.exists():
        d = json.loads(t.read_text())
        methods = d.get("methods", ["raw", "filtered"])
        out["test"] = {"clips": d["clips"], "threshold": d["threshold"], "methods": methods, "llm": d.get("llm"),
                       "scenarios": {k: {m: v[m] for m in methods} for k, v in d["scenarios"].items()}}
    return out


def load_print(speaker: str) -> dict | None:
    p = print_path(speaker)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    d["vector"] = np.array(d["vector"], dtype=np.float32)
    return d


def speech_parts(audio: np.ndarray) -> list[tuple[int, int]]:
    """The speech regions of the audio (sample ranges), from the Silero VAD."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    opts = VadOptions(threshold=0.4, min_speech_duration_ms=150, min_silence_duration_ms=250, speech_pad_ms=100)
    return [(t["start"], t["end"]) for t in get_speech_timestamps(audio, opts, sampling_rate=RATE)]


def enroll(speaker: str, progress=None) -> dict:
    """Make the voice print of the speaker from the training sentences, and set the threshold.

    The threshold is THRESHOLD_SHARE of the median score of the speaker's own validation sentences
    (these sentences are not in the voice print). The noise test showed that a threshold at the
    5% point of the speaker's own scores removes words of the speaker. Other speakers score near 0.
    """
    from .train import collect, selected, split_validation
    from .transcribe import read_wav

    sessions = selected(speaker)
    if not sessions:
        raise ValueError(f"select the sessions of the speaker {speaker} on the training screen")
    clips = [x for x in collect(speaker, sessions=sessions)["train"] if x["part"].startswith("sentences")]
    if len(clips) < 5:
        raise ValueError(f"{speaker}: too few sentences for a voice print ({len(clips)})")
    fit, val = split_validation(clips)
    vectors = []
    for i, x in enumerate(fit):
        audio = read_wav(Path(x["path"]))
        speech = np.concatenate([audio[a:b] for a, b in speech_parts(audio)] or [audio])
        vectors.append(embed(speech))
        if progress:
            progress("enroll", i + 1, len(fit) + len(val))
    vector = np.mean(vectors, axis=0)
    vector /= np.linalg.norm(vector)
    own = []
    for i, x in enumerate(val):
        steps = scores(read_wav(Path(x["path"])), vector)[0]
        own.extend(steps[~np.isnan(steps)].tolist())
        if progress:
            progress("enroll", len(fit) + i + 1, len(fit) + len(val))
    threshold = float(np.median(own)) * THRESHOLD_SHARE
    from datetime import datetime

    data = {"speaker": speaker, "model": SPEAKER_REPO, "created": datetime.now().isoformat(timespec="seconds"),
            "sessions": sessions, "clips": len(fit), "val_clips": len(val),
            "threshold": round(threshold, 4), "own_scores": {
                "p05": round(float(np.quantile(own, 0.05)), 4), "median": round(float(np.median(own)), 4),
                "n": len(own)},
            "vector": [round(float(v), 6) for v in vector]}
    print_path(speaker).parent.mkdir(parents=True, exist_ok=True)
    print_path(speaker).write_text(json.dumps(data, indent=1))
    data["vector"] = vector
    return data


# ---------------------------------------------------------------- filter

def scores(audio: np.ndarray, vector: np.ndarray, window: float = WINDOW) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """A similarity score for each HOP step of the audio (NaN where there is no speech).

    The windows go over the full audio, also over short pauses, because the embedding of a
    short piece of speech is not reliable. A step gets the average score of its windows.
    """
    hop, win = int(HOP * RATE), int(window * RATE)
    n = int(np.ceil(len(audio) / hop)) or 1
    parts = speech_parts(audio)
    mask = np.zeros(len(audio), dtype=bool)
    for a, b in parts:
        mask[a:b] = True
    speech_step = np.array([mask[i * hop:(i + 1) * hop].any() for i in range(n)])
    total, count = np.zeros(n), np.zeros(n)
    if parts and len(audio) <= win:
        total[:] = embed(audio[parts[0][0]:parts[-1][1]]) @ vector
        count[:] = 1
    elif parts:
        for s in range(0, len(audio) - win + 1, hop):
            if mask[s:s + win].mean() >= MIN_SPEECH:
                total[s // hop:(s + win) // hop] += embed(audio[s:s + win]) @ vector
                count[s // hop:(s + win) // hop] += 1
        # A short speech part with no window of enough speech: one embedding of a window around it.
        for a, b in parts:
            if not count[a // hop:int(np.ceil(b / hop))].any():
                c = (a + b) // 2
                lo = max(0, min(c - win // 2, len(audio) - win))
                total[a // hop:int(np.ceil(b / hop))] += embed(audio[lo:lo + win]) @ vector
                count[a // hop:int(np.ceil(b / hop))] += 1
    steps = np.full(n, np.nan)
    has = (count > 0) & speech_step
    steps[has] = total[has] / count[has]
    return steps, parts


def keep_spans(audio: np.ndarray, vector: np.ndarray, threshold: float, window: float = WINDOW,
               steps: tuple | None = None) -> dict:
    """The sample ranges of the audio that have the voice of the speaker.

    `steps` is the result of scores() if the caller has it already.
    """
    hop = int(HOP * RATE)
    steps, parts = steps or scores(audio, vector, window)
    keep = np.nan_to_num(steps, nan=-1) >= threshold
    spans, i = [], 0
    while i < len(keep):
        if keep[i]:
            j = i
            while j < len(keep) and keep[j]:
                j += 1
            if (j - i) * HOP >= MIN_KEEP:
                spans.append((max(0, i * hop - int(PAD * RATE)), min(len(audio), j * hop + int(PAD * RATE))))
            i = j
        else:
            i += 1
    # Cut each span to the speech regions of the VAD, so that the silence stays out.
    cut = []
    for a, b in spans:
        for s, e in parts:
            lo, hi = max(a, s), min(b, e)
            if hi - lo > 0:
                cut.append((lo, hi))
    speech = sum(e - s for s, e in parts) / RATE
    kept = sum(e - s for s, e in cut) / RATE
    return {"spans": cut, "speech_seconds": round(speech, 2), "kept_seconds": round(kept, 2),
            "removed_seconds": round(max(0.0, speech - kept), 2)}


def apply(audio: np.ndarray, speaker: str, threshold: float | None = None,
          vp: dict | None = None) -> tuple[np.ndarray, dict]:
    """Return the audio with only the voice of the speaker, and a report.

    If the speaker has no voice print, return the audio unchanged.
    """
    vp = vp or load_print(speaker)
    if vp is None or len(audio) < RATE * 0.3:
        return audio, {"applied": False}
    threshold = vp["threshold"] if threshold is None else threshold
    info = keep_spans(audio, vp["vector"], threshold)
    return cut_audio(audio, info["spans"]), {"applied": True, "threshold": threshold, **info}


def cut_audio(audio: np.ndarray, spans: list[tuple[int, int]]) -> np.ndarray:
    """Join the kept parts of the audio, with a short silence between them."""
    gap = np.zeros(int(GAP * RATE), dtype=np.float32)
    pieces = []
    for a, b in spans:
        pieces += [audio[a:b], gap]
    return np.concatenate(pieces).astype(np.float32) if pieces else np.zeros(0, dtype=np.float32)


# ---------------------------------------------------------------- word labels

MARGIN = 0.12      # a word with a score below the threshold, but within this distance of it, is "unsure"
GARBLED = 0.35     # a word with a Whisper probability below this, next to another voice, is "unsure"


def label_words(words: list[dict], steps: np.ndarray, threshold: float) -> list[dict]:
    """Give each word of the transcript a label: "own", "other", "unsure", or "silence".

    `words` come from transcribe_words() (start and end in seconds), `steps` from scores().
    A word gets the average score of the HOP steps in its time range. A word in a part with no
    speech (the VAD found none) is "silence": Whisper sometimes writes words where nobody talks.
    """
    out = []
    for w in words:
        i0, i1 = int(w["start"] / HOP), max(int(w["start"] / HOP) + 1, int(np.ceil(w["end"] / HOP)))
        vals = steps[i0:i1]
        vals = vals[~np.isnan(vals)]
        if not len(vals):  # the word times are not exact: look 0.25 s to each side
            near = steps[max(0, i0 - 1):i1 + 1]
            vals = near[~np.isnan(near)]
        score = float(vals.mean()) if len(vals) else float("nan")
        if np.isnan(score):
            label = "silence"
        elif score >= threshold:
            label = "own"  # the language model cannot remove a word with a score above the threshold
        elif score >= threshold - MARGIN:
            label = "unsure"
        else:
            label = "other"
        out.append({**w, "score": None if np.isnan(score) else round(score, 3), "label": label})
    # A garbled word next to another voice can be a mix of the two voices.
    for i, w in enumerate(out):
        near_other = any(0 <= j < len(out) and out[j]["label"] == "other" for j in (i - 1, i + 1))
        if w["label"] == "own" and w.get("prob", 1) < GARBLED and near_other:
            w["label"] = "unsure"
    return out


def join_words(words: list[dict]) -> str:
    """Join word pieces into text. A piece with "join" has no space before it."""
    out = ""
    for w in words:
        out += ("" if w.get("join") and out else " ") + w["word"]
    return out.strip()


def words_text(words: list[dict], keep_unsure: set[int] | None = None) -> str:
    """The text of the words of the speaker. Keep the unsure words with an index in keep_unsure
    (None: keep all the unsure words)."""
    return join_words([w for i, w in enumerate(words) if w["label"] == "own"
                       or (w["label"] == "unsure" and (keep_unsure is None or i in keep_unsure))])


def soft_cut(audio: np.ndarray, vector: np.ndarray, threshold: float) -> tuple[np.ndarray, tuple, dict]:
    """Remove only the parts that are surely another voice (a score below threshold - MARGIN).

    Whisper often writes only one of two speakers, so it must not hear a clear other voice.
    The parts near the threshold stay in the audio: the word labels decide about them.
    Returns (the cut audio, the scores of the original audio, a report with the kept spans).
    """
    steps = scores(audio, vector)
    info = keep_spans(audio, vector, threshold - MARGIN, steps=steps)
    return cut_audio(audio, info["spans"]), steps, info


def to_original(t: float, spans: list[tuple[int, int]]) -> float:
    """Change a time in the cut audio (seconds) to the time in the original audio.

    Do not score the cut audio again: a window over the joint of two kept parts mixes them.
    """
    pos = 0.0
    for a, b in spans:
        length = (b - a) / RATE
        if t <= pos + length:
            return a / RATE + max(0.0, t - pos)
        pos += length + GAP
        if t < pos:  # in the silence between two parts
            return b / RATE
    return spans[-1][1] / RATE if spans else t


def filter_words(model, audio: np.ndarray, vp: dict) -> tuple[list[dict], dict]:
    """The "words" mode: soft cut, then transcribe with word times, then give each word a label."""
    from .transcribe import transcribe_words

    cut, (steps, _), info = soft_cut(audio, vp["vector"], vp["threshold"])
    if len(cut) < RATE * 0.3:
        return [], info
    words = [{**w, "start": to_original(w["start"], info["spans"]), "end": to_original(w["end"], info["spans"])}
             for w in transcribe_words(model, cut)]
    labels = label_words(words, steps, vp["threshold"])
    # The soft cut removed the audio that is surely another voice. A word in the kept audio with a
    # low score is often a word of the speaker next to the other voice: the windows of 1.5 s
    # spread the low score over it, and the word times at a joint are not exact. So a word gets
    # the highest score in its time range and one step to each side, and only a very low score
    # is "other" here.
    for w in labels:
        if w["label"] in ("other", "unsure"):
            i0, i1 = int(w["start"] / HOP) - 1, int(np.ceil(w["end"] / HOP)) + 1
            near = steps[max(0, i0):i1]
            near = near[~np.isnan(near)]
            best = float(near.max()) if len(near) else w["score"]
            if best is not None and best >= vp["threshold"]:
                w["label"] = "own"
            elif best is not None and best >= vp["threshold"] - 2 * MARGIN:
                w["label"] = "unsure"
    return labels, info


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--enroll", metavar="SPEAKER")
    p.add_argument("--test", metavar="SPEAKER")
    p.add_argument("--limit", type=int, default=0, help="the test: use only the first N test clips")
    a = p.parse_args()
    if a.enroll:
        d = enroll(a.enroll)
        print({k: v for k, v in d.items() if k != "vector"})
    if a.test:
        from .voice_test import run

        run(a.test, a.limit)


if __name__ == "__main__":
    main()
