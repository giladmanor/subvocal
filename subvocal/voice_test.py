"""The noise test of the voice filter (subvocal/voice.py).

The test mixes each test-set clip of the speaker with the speech of other Hebrew speakers,
and compares these methods:

    raw           no filter
    cut           remove the parts of the audio with another voice, then transcribe (voice.apply)
    words         remove only the parts that are surely another voice, transcribe the rest with
                  word times, give each word a label, and remove the words of other people.
                  Keep the unsure words. (voice.filter_words)
    words+llm     the same, and the language model decides about the unsure words (llm.decide)

Other speakers: the dev part of Google FLEURS, Hebrew (CC-BY-4.0), in data/test-audio/fleurs-he/.
These recordings come from other microphones and rooms. A real person near the speaker comes
through the same microphone, so the real results can be worse than the results of this test.

The scenarios:
    clean         the clip without other speech (the filter must not remove the speaker's words)
    around -6dB   another person talks before and after the speaker, 6 dB quieter (farther away)
    around 0dB    the same, at the same level as the speaker
    between       another person talks in the longest pause of the speaker, 6 dB quieter
    overlap 10dB  another person talks at the same time as the speaker, 10 dB quieter
    overlap 0dB   the same, at the same level (the filter cannot separate this)

The measures:
    WER           word error rate against the label
    extra         words in the transcript that the speaker did not say (insertions)
    lost          words of the speaker that are not in the transcript (deletions)
    other kept    the share of the other person's speech that the filter did not remove
    own kept      the share of the speaker's speech that the filter kept

    python -m subvocal.voice --test gilad [--limit 10]
"""

from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

import numpy as np

from . import settings, voice
from .numbers import canonical
from .phonetics import REPO_ROOT
from .diagnose import normalize

FLEURS = REPO_ROOT / "data" / "test-audio" / "fleurs-he"
RATE = voice.RATE
SWEEP = (0.2, 0.3, 0.4, 0.5, 0.6)  # thresholds for the frame measures (no transcription)
SCENARIOS = ("clean", "around -6dB", "around 0dB", "between", "overlap 10dB", "overlap 0dB")


def download_fleurs() -> None:
    """Download the Hebrew dev part of FLEURS (153 MB) one time."""
    import tarfile

    from huggingface_hub import hf_hub_download

    if (FLEURS / "audio" / "dev").exists():
        return
    for f in ("data/he_il/dev.tsv", "data/he_il/audio/dev.tar.gz"):
        path = hf_hub_download("google/fleurs", f, repo_type="dataset", local_dir=FLEURS)
    with tarfile.open(path) as tar:
        tar.extractall(FLEURS / "audio", filter="data")


def others() -> list[np.ndarray]:
    import soundfile as sf

    download_fleurs()
    rows = list(csv.reader(open(FLEURS / "data" / "he_il" / "dev.tsv", encoding="utf-8"), delimiter="\t"))
    out = []
    for r in rows:
        audio, rate = sf.read(FLEURS / "audio" / "dev" / r[1], dtype="float32")
        if rate == RATE and len(audio) > RATE * 2:
            out.append(audio)
    return out


def speech_rms(audio: np.ndarray) -> float:
    parts = voice.speech_parts(audio)
    speech = np.concatenate([audio[a:b] for a, b in parts]) if parts else audio
    return float(np.sqrt(np.mean(speech ** 2)) + 1e-9)


def ops(ref: str, hyp: str) -> dict:
    """Count the word errors by type: substitutions, insertions (extra), deletions (lost)."""
    r, h = canonical(normalize(ref)).split(), canonical(normalize(hyp)).split()
    d = [[(j, 0, j, 0) for j in range(len(h) + 1)]]  # (cost, subs, ins, dels)
    for i in range(1, len(r) + 1):
        row = [(i, 0, 0, i)]
        for j in range(1, len(h) + 1):
            same = r[i - 1] == h[j - 1]
            a, b, c = d[i - 1][j - 1], d[i - 1][j], row[j - 1]
            row.append(min((a[0] + (not same), a[1] + (not same), a[2], a[3]),
                           (b[0] + 1, b[1], b[2], b[3] + 1),
                           (c[0] + 1, c[1], c[2] + 1, c[3])))
        d.append(row)
    cost, subs, ins, dels = d[-1][-1]
    return {"errors": cost, "words": len(r), "subs": subs, "extra": ins, "lost": dels}


def mix(target: np.ndarray, kind: str, pool: list[np.ndarray], rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Make one mix. Returns (audio, label) where the label is 1 for the speaker and 2 for the other person."""
    t_rms = speech_rms(target)
    other = pool[rng.randrange(len(pool))]

    def piece(seconds: float, db: float) -> np.ndarray:
        n = int(seconds * RATE)
        start = rng.randrange(max(1, len(other) - n))
        p = other[start:start + n]
        return (p * (t_rms * 10 ** (db / 20) / speech_rms(p))).astype(np.float32)

    t_label = np.zeros(len(target), dtype=np.int8)
    for a, b in voice.speech_parts(target):
        t_label[a:b] = 1
    gap = np.zeros(int(0.3 * RATE), dtype=np.float32)
    if kind == "clean":
        return target, t_label
    if kind.startswith("around"):
        db = float(kind.split()[1].replace("dB", ""))
        p1, p2 = piece(2.0, db), piece(2.0, db)
        audio = np.concatenate([p1, gap, target, gap, p2])
        label = np.concatenate([np.full(len(p1), 2), np.zeros(len(gap)), t_label, np.zeros(len(gap)),
                                np.full(len(p2), 2)]).astype(np.int8)
        return audio, label
    if kind == "between":
        parts = voice.speech_parts(target)
        pauses = [(parts[i][1], parts[i + 1][0]) for i in range(len(parts) - 1)]
        cut = (max(pauses, key=lambda p: p[1] - p[0])[0] + max(pauses, key=lambda p: p[1] - p[0])[1]) // 2 \
            if pauses else len(target) // 2
        p = piece(1.5, -6)
        audio = np.concatenate([target[:cut], gap, p, gap, target[cut:]])
        label = np.concatenate([t_label[:cut], np.zeros(len(gap)), np.full(len(p), 2), np.zeros(len(gap)),
                                t_label[cut:]]).astype(np.int8)
        return audio, label
    db = -float(kind.split()[1].replace("dB", ""))  # overlap
    p = piece(len(target) / RATE, db)
    p = np.pad(p, (0, len(target) - len(p)))
    label = t_label.copy()
    label[(t_label == 0) & (np.abs(p) > 0)] = 2
    return (target + p).astype(np.float32), label


METHODS = ("raw", "cut", "words", "words+llm")


def kept_shares(audio: np.ndarray, label: np.ndarray, vector: np.ndarray, threshold: float,
                steps: tuple) -> tuple[float, float]:
    keep = np.zeros(len(audio), dtype=bool)
    for a, b in voice.keep_spans(audio, vector, threshold, steps=steps)["spans"]:
        keep[a:b] = True
    own, other = label == 1, label == 2
    return (float(keep[own].mean()) if own.any() else float("nan"),
            float(keep[other].mean()) if other.any() else float("nan"))


def run(speaker: str, limit: int = 0) -> dict:
    from .llm import decide
    from .train import collect, selected
    from .transcribe import load_model, read_wav, transcribe_audio
    from .trial import Trial

    vp = voice.load_print(speaker) or voice.enroll(speaker)
    thr = vp["threshold"]
    model_path, model_label = Trial().model_path(speaker)
    model = load_model(model_path, cpu_threads=6)
    llm_model = settings.load()["llm"]["model"]
    clips = [x for x in collect(speaker, sessions=selected(speaker) or None)["test"]
             if np.abs(read_wav(Path(x["path"]))).max() >= 0.02]
    clips = clips[:limit] if limit else clips
    pool = others()
    print(f"speaker {speaker}: {len(clips)} test clips, {len(pool)} clips of other speakers, "
          f"model {model_label}, threshold {thr}, language model {llm_model}", flush=True)

    rows = {k: {m: [] for m in METHODS} | {"own": {t: [] for t in SWEEP}, "other": {t: [] for t in SWEEP},
                                           "own_cal": [], "other_cal": [], "llm_calls": 0, "unsure": 0}
            for k in SCENARIOS}
    items = []
    t0 = time.time()
    for i, x in enumerate(clips):
        target = read_wav(Path(x["path"]))
        rng = random.Random(x["id"])
        for kind in SCENARIOS:
            audio, label = mix(target, kind, pool, rng)
            steps = voice.scores(audio, vp["vector"])
            cut = voice.cut_audio(audio, voice.keep_spans(audio, vp["vector"], thr, steps=steps)["spans"])
            labels = voice.filter_words(model, audio, vp)[0]
            unsure = sum(w["label"] == "unsure" for w in labels)
            r = rows[kind]
            r["unsure"] += unsure
            if unsure:
                r["llm_calls"] += 1
            hyps = {
                "raw": transcribe_audio(model, audio),
                "cut": transcribe_audio(model, cut) if len(cut) else "",
                "words": voice.words_text(labels),
                "words+llm": voice.words_text(labels, decide(labels, [], llm_model)) if unsure
                else voice.words_text(labels, set()),
            }
            for m in METHODS:
                r[m].append(ops(x["text"], hyps[m]))
            own, oth = kept_shares(audio, label, vp["vector"], thr, steps)
            r["own_cal"].append(own)
            r["other_cal"].append(oth)
            for t in SWEEP:
                own, oth = kept_shares(audio, label, vp["vector"], t, steps)
                r["own"][t].append(own)
                r["other"][t].append(oth)
            items.append({"id": x["id"], "scenario": kind, "ref": x["text"], **hyps,
                          "labels": [{k: w[k] for k in ("word", "label", "score", "prob")} for w in labels]})
        print(f"  {i + 1}/{len(clips)}  ({time.time() - t0:.0f} s)", flush=True)

    def total(ops_list: list[dict]) -> dict:
        s = {k: sum(o[k] for o in ops_list) for k in ("errors", "words", "subs", "extra", "lost")}
        return {"wer": round(s["errors"] / s["words"], 4) if s["words"] else None, **s}

    mean = lambda xs: round(float(np.nanmean(xs)), 3) if not all(np.isnan(xs)) else None  # noqa: E731
    report = {"speaker": speaker, "clips": len(clips), "model": model_label, "threshold": thr,
              "speaker_model": voice.SPEAKER_REPO, "llm": llm_model, "methods": list(METHODS), "scenarios": {}}
    for kind, r in rows.items():
        report["scenarios"][kind] = {
            **{m: total(r[m]) for m in METHODS},
            "own_kept": mean(r["own_cal"]), "other_kept": mean(r["other_cal"]),
            "llm_calls": r["llm_calls"], "unsure_words": r["unsure"],
            "sweep": {str(t): {"own_kept": mean(r["own"][t]), "other_kept": mean(r["other"][t])} for t in SWEEP},
        }
    out = settings.MODELS / speaker / "voice-test.json"
    out.write_text(json.dumps({**report, "items": items}, ensure_ascii=False, indent=1))
    print_report(report)
    print(f"\nwrote {out}")
    return report


def print_report(r: dict) -> None:
    fmt = lambda v: "—" if v is None or v != v else f"{v:.0%}"  # noqa: E731
    print(f"\nthreshold {r['threshold']}, language model {r['llm']}\n")
    print("WER / extra words / lost words")
    print(f"{'scenario':14} " + " ".join(f"{m:>18}" for m in r["methods"]) + "   llm calls")
    for kind, s in r["scenarios"].items():
        print(f"{kind:14} " + " ".join(f"{fmt(s[m]['wer']):>6} {s[m]['extra']:>4} {s[m]['lost']:>5}  " for m in r["methods"])
              + f"   {s['llm_calls']} ({s['unsure_words']} unsure words)")
    print("\nthreshold sweep for the cut method (own kept / other kept):")
    for kind, s in r["scenarios"].items():
        print(f"  {kind:14} " + "  ".join(f"{t}: {v['own_kept'] or 0:.0%}/{(v['other_kept'] or 0):.0%}"
                                          for t, v in s["sweep"].items()))
