"""Make the sentence list of a training session from its word test.

1. The script makes the speaker profile from the word test of the session only.
   Each session starts from the beginning.
2. It selects the sentences that the speaker did not record in an earlier session.
   The selection gives more occurrences to the sounds with the highest error rate
   (see subvocal/select.py).
3. It writes profile.json and sentence-list.tsv in the session folder.

Usage:
    python -m subvocal.plan --session dana-2026-11-23-092316 --n 40
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .corpus import SENTENCES_TSV, read_tsv, write_tsv
from .diagnose import error_rates, merge, run
from .groups import LIST_FILE, group_dir, list_groups
from .phonetics import Phonemizer, flatten, split_text
from .review import RECORDINGS
from .select import select

# A Hebrew letter or a vowel name for each phoneme, for the app screen.
HEBREW = {
    "p": "פּ", "b": "בּ", "t": "ת/ט", "d": "ד", "k": "כּ/ק", "ɡ": "ג", "ʔ": "א/ע", "f": "פ", "v": "ב/ו",
    "s": "ס/שׂ", "z": "ז", "ʃ": "שׁ", "ʒ": "ז׳", "χ": "ח/כ", "ʁ": "ר", "h": "ה", "ts": "צ", "tʃ": "צ׳",
    "dʒ": "ג׳", "m": "מ", "n": "נ", "l": "ל", "j": "י", "w": "ו (w)",
    "a": "תנועת a", "e": "תנועת e", "i": "תנועת i", "o": "תנועת o", "u": "תנועת u",
}

_PHONEMIZER: Phonemizer | None = None


def word_profile(folder: Path) -> dict:
    """The profile from the transcribed word test of one session."""
    global _PHONEMIZER
    words = folder / "words"
    if not (words / "hyp.tsv").exists():
        raise ValueError("the word test is not transcribed yet")
    _PHONEMIZER = _PHONEMIZER or Phonemizer()
    refs = read_tsv(words / "prompts.tsv")
    hyps = {r["id"]: r["text"] for r in read_tsv(words / "hyp.tsv")}
    return merge({}, run(refs, hyps, True, _PHONEMIZER), f"{folder.name}/words")


def recorded_before(folder: Path, speaker: str) -> list[str]:
    """The sentence IDs that the speaker recorded in the other sessions."""
    done: list[str] = []
    for g in list_groups():
        if g["speaker"] == speaker and g["name"] != folder.name:
            done += [w.stem for w in sorted((RECORDINGS / g["name"] / "sentences").glob("*.wav"))]
    return list(dict.fromkeys(done))


def _sounds(profile: dict, rates: dict, top: int = 8) -> list[dict]:
    seen = profile.get("phonemes", {})
    best = sorted((p for p in rates if seen.get(p, {}).get("n")), key=rates.get, reverse=True)[:top]
    return [{"phoneme": p, "hebrew": HEBREW.get(p, p), "rate": round(rates[p], 3),
             "n": seen[p]["n"], "err": seen[p]["err"]} for p in best]


def session_sounds(name: str) -> dict:
    """The problem sounds from the word test. This is fast, and does not select sentences."""
    profile = word_profile(group_dir(name))
    return {"name": name, "sounds": _sounds(profile, error_rates(profile))}


def make_list(name: str, n: int = 40, keep: bool = False) -> dict:
    """Select the sentences and write the list. With keep=True, use the list that exists."""
    folder = group_dir(name)
    speaker = json.loads((folder / "group.json").read_text())["speaker"]
    profile = word_profile(folder)
    rates = error_rates(profile)
    weights = {p: 0.1 + r for p, r in rates.items()}
    sentences = read_tsv(SENTENCES_TSV)
    done = recorded_before(folder, speaker)
    if keep and (folder / LIST_FILE).exists():
        chosen = read_tsv(folder / LIST_FILE)
        n = len(chosen)
    else:
        chosen = select(sentences, weights, n, done, rates)
        write_tsv(folder / LIST_FILE, chosen, ["id", "section", "text", "niqqud", "phonemes", "gain"])
    (folder / "profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=1))

    # How many times each problem sound occurs in the new list, and in a plain list of the same size.
    def counts(rows):
        return Counter(p for r in rows for p in flatten(split_text(r["phonemes"])))
    plain = select(sentences, {p: 1.0 for p in rates}, n, done)
    have, base = counts(chosen), counts(plain)
    return {
        "name": name, "count": len(chosen), "recorded_before": len(done),
        "remaining": len(sentences) - len(done),
        "sounds": [{**x, "in_list": have[x["phoneme"]], "in_plain": base[x["phoneme"]]}
                   for x in _sounds(profile, rates)],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True, help="the session folder name in data/user-recordings")
    ap.add_argument("--n", type=int, default=40, help="number of sentences")
    args = ap.parse_args()
    plan = make_list(args.session, args.n)
    print(f"Wrote {plan['count']} sentences to {RECORDINGS / args.session / LIST_FILE}")
    print(f"{'sound':>6} {'err':>4} {'n':>4} {'rate':>5} {'in list':>8} {'plain':>6}")
    for s in plan["sounds"]:
        print(f"{s['phoneme']:>6} {s['err']:>4} {s['n']:>4} {s['rate']:>5.0%} {s['in_list']:>8} {s['in_plain']:>6}")


if __name__ == "__main__":
    main()
