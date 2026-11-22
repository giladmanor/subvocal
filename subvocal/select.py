"""Select the next sentences for a speaker to record.

The selection is greedy. At each step, the script adds the sentence with the highest gain:

    gain(sentence) = sum over its phonemes p of  weight[p] * decay ** (have[p] / scale[p])
                     + pair_bonus for each phoneme pair that the set does not have yet
                     divided by sqrt(number of phonemes in the sentence)

- weight[p] comes from the speaker profile: 0.1 + the smoothed error rate of p.
  With no profile, all weights are 1, and the result is a plain coverage set.
- have[p] is the count of p in the sentences that are already selected or recorded.
  The decay gives less value to each new occurrence of a phoneme.
- scale[p] is 1 + 10 * the error rate of p. A problem sound loses value more slowly,
  so the set gets many more occurrences of it.
- The division by the length prefers short sentences that contain many target sounds.

Usage:
    python -m subvocal.select --n 40                                  # first session, no profile
    python -m subvocal.select --n 40 --profile data/dana/profile.json \
        --done data/dana/recorded.txt --out data/dana/next.tsv
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from .corpus import SENTENCES_TSV, read_tsv, write_tsv
from .coverage import report
from .diagnose import error_rates
from .phonetics import PHONEMES, flatten, split_text


def sentence_features(row: dict) -> tuple[Counter, set]:
    words = split_text(row["phonemes"])
    return Counter(flatten(words)), {(a, b) for w in words for a, b in zip(w, w[1:])}


def select(rows, weights, n, done_ids=(), rates=None, decay=0.85, pair_bonus=0.05):
    rates = rates or {}
    scale = {p: 1 + 10 * rates.get(p, 0.0) for p in PHONEMES}
    feats = {r["id"]: sentence_features(r) for r in rows}
    have: Counter = Counter()
    pairs: set = set()
    for i in done_ids:
        if i in feats:
            have.update(feats[i][0])
            pairs |= feats[i][1]
    pool = [r for r in rows if r["id"] not in set(done_ids)]
    chosen = []
    while pool and len(chosen) < n:
        def gain(r):
            counts, prs = feats[r["id"]]
            g = sum(weights.get(p, 0) * sum(decay ** ((have[p] + k) / scale[p]) for k in range(c))
                    for p, c in counts.items())
            g += pair_bonus * len(prs - pairs)
            return g / math.sqrt(max(1, sum(counts.values())))

        best = max(pool, key=gain)
        chosen.append({**best, "gain": f"{gain(best):.3f}"})
        have.update(feats[best["id"]][0])
        pairs |= feats[best["id"]][1]
        pool.remove(best)
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sentences", type=Path, default=SENTENCES_TSV)
    ap.add_argument("--profile", type=Path, help="speaker profile JSON from subvocal.diagnose")
    ap.add_argument("--done", type=Path, help="file with the IDs that the speaker already recorded")
    ap.add_argument("--n", type=int, default=40, help="number of sentences to select")
    ap.add_argument("--out", type=Path, help="write the selection to this TSV")
    args = ap.parse_args()

    rows = read_tsv(args.sentences)
    rates = None
    if args.profile:
        rates = error_rates(json.loads(args.profile.read_text()))
        weights = {p: 0.1 + r for p, r in rates.items()}
    else:
        weights = {p: 1.0 for p in PHONEMES}
    done = [l.split()[0] for l in args.done.read_text().splitlines() if l.strip()] if args.done else []

    chosen = select(rows, weights, args.n, done, rates)
    for r in chosen:
        print(f"{r['id']}  {r['gain']}  {r['text']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_tsv(args.out, chosen, ["id", "section", "text", "phonemes", "gain"])
        print(f"\nWrote {len(chosen)} sentences to {args.out}")
    print()
    report(chosen, minimum=0)


if __name__ == "__main__":
    main()
