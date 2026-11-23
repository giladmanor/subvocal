"""Find the sounds that the transcription model gets wrong for one speaker.

Inputs:
- a reference file: prompts/probe.tsv (the diagnostic list) or prompts/sentences.tsv;
- a hypothesis file: a TSV with the columns "id" and "text" (the output of subvocal.transcribe).

The script aligns the reference phonemes with the phonemes of the transcript. It counts the
errors for each phoneme and writes a speaker profile (JSON). If the profile file exists, the
script adds the new counts to it. Thus, the profile improves after each recording session.

Usage:
    python -m subvocal.diagnose --hyp data/dana/probe/hyp.tsv --profile data/dana/profile.json
    python -m subvocal.diagnose --refs prompts/sentences.tsv --hyp data/dana/s1/hyp.tsv \
        --profile data/dana/profile.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from .corpus import PROBE_TSV, read_tsv
from .numbers import canonical
from .phonetics import PHONEMES, Phonemizer, align, flatten, split_text

CARRIER = {"המילה", "היא", "מילה"}
# Phonemes of the carrier phrase "המילה היא". Diagnosis aligns the full phrase, because the
# model sometimes joins the carrier and the word (for example "המילה הסתיים").
CARRIER_PHONEMES = "h a m i l a h i".split()
_NIQQUD = re.compile(r"[֑-ׇ]")
_PUNCT = re.compile(r"[^\w\s׳]")


def normalize(text: str) -> str:
    text = _NIQQUD.sub("", text)
    for ch in "'’`׳":
        text = text.replace(ch, "׳")
    return " ".join(_PUNCT.sub(" ", text).split())


def strip_carrier(text: str) -> str:
    return " ".join(t for t in normalize(text).split() if t not in CARRIER)


def score_item(ref_phonemes: list[str], hyp_phonemes: list[str], stats: dict, skip: int = 0) -> list[str]:
    """Add the alignment of one item to the stats. Return the phonemes that had an error.

    The first `skip` reference phonemes (the carrier phrase) are aligned but not counted.
    """
    wrong = []
    ref_index = 0
    for r, h in align(ref_phonemes, hyp_phonemes):
        if r is not None:
            ref_index += 1
        if ref_index <= skip:
            continue
        if r is None:
            stats["insertions"][h] += 1
            continue
        stats["n"][r] += 1
        if h != r:
            stats["err"][r] += 1
            stats["confusions"][f"{r}>{h or '-'}"] += 1
            wrong.append(r)
    return wrong


def run(refs: list[dict], hyps: dict[str, str], is_probe: bool, ph: Phonemizer) -> dict:
    stats = {k: Counter() for k in ("n", "err", "confusions", "insertions")}
    items = []
    for ref in refs:
        if ref["id"] not in hyps:
            continue
        hyp_text = hyps[ref["id"]]
        if is_probe:
            ref_ph = CARRIER_PHONEMES + ref["phonemes"].split()
            hyp_norm = strip_carrier(hyp_text)
            accepted = {ref["word"], *normalize(ref.get("accept") or "").split()}
            words = hyp_norm.split()
            ok = len(words) == 1 and words[0] in accepted  # extra sounds count as errors
            full = normalize(hyp_text)
            hyp_ph = ref_ph if ok else flatten(split_text(ph(full))) if full else []
            foils = [f for f in ref["foils"].split() if f in hyp_norm.split()]
            skip = len(CARRIER_PHONEMES)
        else:
            ref_ph = flatten(split_text(ref["phonemes"]))
            hyp_norm = normalize(hyp_text)
            ok = canonical(hyp_norm) == canonical(normalize(ref["text"]))  # "97" = "תשעים ושבע"
            hyp_ph = ref_ph if ok else flatten(split_text(ph(hyp_norm))) if hyp_norm else []
            foils = []
            skip = 0
        wrong = score_item(ref_ph, hyp_ph, stats, skip)
        if wrong:
            items.append({"id": ref["id"], "ref": ref.get("word") or ref["text"],
                          "hyp": hyp_norm, "wrong": wrong, "foils": foils})
    return {"stats": stats, "items": items}


def merge(profile: dict, result: dict, source: str) -> dict:
    ph = profile.setdefault("phonemes", {})
    for p in result["stats"]["n"]:
        e = ph.setdefault(p, {"n": 0, "err": 0})
        e["n"] += result["stats"]["n"][p]
        e["err"] += result["stats"]["err"][p]
    conf = profile.setdefault("confusions", {})
    for k, v in result["stats"]["confusions"].items():
        conf[k] = conf.get(k, 0) + v
    profile.setdefault("sessions", []).append({"source": source, "errors": result["items"]})
    return profile


def error_rates(profile: dict, strength: float = 2.0) -> dict[str, float]:
    """Smoothed error rate for each phoneme.

    A phoneme with few observations moves toward the mean error rate of the speaker.
    This prevents one error in one item from giving a rate of 100%.
    """
    ph = profile.get("phonemes", {})
    n = sum(e["n"] for e in ph.values())
    prior = (sum(e["err"] for e in ph.values()) / n) if n else 0.0
    return {p: (ph.get(p, {}).get("err", 0) + strength * prior)
               / (ph.get(p, {}).get("n", 0) + strength)
            for p in PHONEMES}


def print_report(profile: dict, top: int = 12) -> None:
    rates = error_rates(profile)
    ph = profile.get("phonemes", {})
    print(f"{'phoneme':>8} {'n':>5} {'err':>5} {'rate':>6}")
    for p in sorted(rates, key=rates.get, reverse=True)[:top]:
        e = ph.get(p, {"n": 0, "err": 0})
        print(f"{p:>8} {e['n']:>5} {e['err']:>5} {rates[p]:>6.0%}")
    conf = sorted(profile.get("confusions", {}).items(), key=lambda kv: -kv[1])[:top]
    if conf:
        print("Top confusions (reference>transcript): " + ", ".join(f"{k}:{v}" for k, v in conf))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refs", type=Path, default=PROBE_TSV, help="probe.tsv or sentences.tsv")
    ap.add_argument("--hyp", type=Path, required=True, help="TSV with columns id, text")
    ap.add_argument("--profile", type=Path, required=True, help="speaker profile JSON (created or updated)")
    args = ap.parse_args()

    refs = read_tsv(args.refs)
    is_probe = "word" in refs[0]
    hyps = {r["id"]: r["text"] for r in read_tsv(args.hyp)}
    result = run(refs, hyps, is_probe, Phonemizer())

    profile = json.loads(args.profile.read_text()) if args.profile.exists() else {}
    merge(profile, result, str(args.hyp))
    args.profile.parent.mkdir(parents=True, exist_ok=True)
    args.profile.write_text(json.dumps(profile, ensure_ascii=False, indent=1))

    print(f"Items with errors: {len(result['items'])} of {len(hyps)}")
    for it in result["items"]:
        foil = f"  (foil: {' '.join(it['foils'])})" if it["foils"] else ""
        print(f"  {it['id']}: {it['ref']} -> {it['hyp'] or '(empty)'}  wrong: {' '.join(it['wrong'])}{foil}")
    print()
    print_report(profile)
    print(f"\nProfile written to {args.profile}")


if __name__ == "__main__":
    main()
