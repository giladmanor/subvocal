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


def make_plan(speaker: str, n: int = 40) -> dict:
    folders = speaker_sessions(speaker)
    if not folders:
        raise ValueError(f"no transcribed sessions for the speaker: {speaker}")
    ph = Phonemizer()
    profile: dict = {}
    done: list[str] = []
    for folder in folders:
        refs = read_tsv(folder / "prompts.tsv")
        hyps = {r["id"]: r["text"] for r in read_tsv(folder / "hyp.tsv")}
        is_probe = "word" in refs[0]
        merge(profile, run(refs, hyps, is_probe, ph), folder.name)
        if not is_probe:
            done += [i for i in hyps if i not in done]

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
    seen = profile.get("phonemes", {})
    top = sorted((p for p in rates if seen.get(p, {}).get("n")), key=rates.get, reverse=True)[:8]
    return {
        "speaker": speaker, "list": str(list_path.relative_to(RECORDINGS.parent.parent)),
        "count": len(chosen), "sessions": [f.name for f in folders], "recorded": len(done),
        "remaining": len(sentences) - len(done),
        "sounds": [{"phoneme": p, "hebrew": HEBREW.get(p, p), "rate": round(rates[p], 3),
                    "n": seen[p]["n"], "err": seen[p]["err"], "in_list": have[p], "in_plain": base[p]}
                   for p in top],
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
