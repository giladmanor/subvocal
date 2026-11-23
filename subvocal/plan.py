"""Make the next sentence list for a speaker from the sound test and the earlier sessions.

The script uses all the transcribed sessions of the speaker (probe and sentences):

1. It builds the speaker profile again from all of these sessions. Thus, a session
   that you transcribe again is not counted two times.
2. It selects the sentences that the speaker did not record yet. The selection gives
   more occurrences to the sounds with the highest error rate (see subvocal/select.py).
3. It writes data/user-recordings/<speaker>-profile.json and <speaker>-next.tsv.
   The recorder shows the TSV in its list menu.

Usage:
    python -m subvocal.plan --speaker dana --n 40
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from .corpus import SENTENCES_TSV, read_tsv, write_tsv
from .diagnose import error_rates, merge, run
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


def speaker_sessions(speaker: str) -> list:
    out = []
    for folder in sorted(RECORDINGS.iterdir()) if RECORDINGS.exists() else []:
        meta_path = folder / "session.json"
        if not (meta_path.exists() and (folder / "hyp.tsv").exists() and (folder / "prompts.tsv").exists()):
            continue
        if json.loads(meta_path.read_text()).get("speaker") == speaker:
            out.append(folder)
    return out


_PHONEMIZER: Phonemizer | None = None


def build_profile(speaker: str) -> tuple[dict, list[str], list[str]]:
    """The profile from all the transcribed sessions, the session names, and the recorded sentence IDs."""
    global _PHONEMIZER
    folders = speaker_sessions(speaker)
    if not folders:
        raise ValueError(f"no transcribed sessions for the speaker: {speaker}")
    _PHONEMIZER = _PHONEMIZER or Phonemizer()
    profile: dict = {}
    done: list[str] = []
    for folder in folders:
        refs = read_tsv(folder / "prompts.tsv")
        hyps = {r["id"]: r["text"] for r in read_tsv(folder / "hyp.tsv")}
        is_probe = "word" in refs[0]
        merge(profile, run(refs, hyps, is_probe, _PHONEMIZER), folder.name)
        if not is_probe:
            done += [i for i in hyps if i not in done]
    return profile, [f.name for f in folders], done


def _sounds(profile: dict, rates: dict, top: int = 8) -> list[dict]:
    seen = profile.get("phonemes", {})
    best = sorted((p for p in rates if seen.get(p, {}).get("n")), key=rates.get, reverse=True)[:top]
    return [{"phoneme": p, "hebrew": HEBREW.get(p, p), "rate": round(rates[p], 3),
             "n": seen[p]["n"], "err": seen[p]["err"]} for p in best]


def profile_sounds(speaker: str) -> dict:
    """The problem sounds of the speaker. This is fast, and does not select sentences."""
    profile, sessions, done = build_profile(speaker)
    return {"speaker": speaker, "sessions": sessions, "recorded": len(done),
            "remaining": len(read_tsv(SENTENCES_TSV)) - len(done),
            "sounds": _sounds(profile, error_rates(profile))}


def make_plan(speaker: str, n: int = 40) -> dict:
    profile, sessions, done = build_profile(speaker)
    rates = error_rates(profile)
    weights = {p: 0.1 + r for p, r in rates.items()}
    sentences = read_tsv(SENTENCES_TSV)
    chosen = select(sentences, weights, n, done, rates)

    profile_path = RECORDINGS / f"{speaker}-profile.json"
    list_path = RECORDINGS / f"{speaker}-next.tsv"
    profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=1))
    write_tsv(list_path, chosen, ["id", "section", "text", "niqqud", "phonemes", "gain"])

    # How many times each problem sound occurs in the new list, and in a plain list of the same size.
    def counts(rows):
        return Counter(p for r in rows for p in flatten(split_text(r["phonemes"])))
    plain = select(sentences, {p: 1.0 for p in rates}, n, done)
    have, base = counts(chosen), counts(plain)
    sounds = [{**x, "in_list": have[x["phoneme"]], "in_plain": base[x["phoneme"]]} for x in _sounds(profile, rates)]
    return {
        "speaker": speaker, "list": str(list_path.relative_to(RECORDINGS.parent.parent)),
        "count": len(chosen), "sessions": sessions, "recorded": len(done),
        "remaining": len(sentences) - len(done), "sounds": sounds,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speaker", required=True)
    ap.add_argument("--n", type=int, default=40, help="number of sentences")
    args = ap.parse_args()
    plan = make_plan(args.speaker, args.n)
    print(f"Sessions: {', '.join(plan['sessions'])}")
    print(f"Wrote {plan['count']} sentences to {plan['list']}")
    print(f"{'sound':>6} {'err':>4} {'n':>4} {'rate':>5} {'in list':>8} {'plain':>6}")
    for s in plan["sounds"]:
        print(f"{s['phoneme']:>6} {s['err']:>4} {s['n']:>4} {s['rate']:>5.0%} {s['in_list']:>8} {s['in_plain']:>6}")


if __name__ == "__main__":
    main()
