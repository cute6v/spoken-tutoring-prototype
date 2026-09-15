# recompute_wer.py
# Re-computes WER from an existing wer_results.csv using Whisper's standard
# English text normalizer + digit-run collapsing, so number-format mismatches
# (e.g. "two six four eight" vs "2 6 4 8") are not miscounted as errors.
# Does NOT re-run Whisper. Reads results/wer_results.csv, writes results/wer_results_v2.csv.

import re
import csv
import jiwer
from whisper.normalizers import EnglishTextNormalizer

_base = EnglishTextNormalizer()

def normalize(text):
    t = _base(text or "")
    t = re.sub(r'(?<=\d)\s+(?=\d)', '', t)   # "2 6 4 8" -> "2648"
    return t.strip()

def is_number_utt(ref_norm):
    return bool(re.fullmatch(r'\d+', ref_norm.replace(' ', '')))

IN, OUT = "results/wer_results.csv", "results/wer_results_v2.csv"

rows = list(csv.DictReader(open(IN, encoding="utf-8")))
out = []
for r in rows:
    rn, hn = normalize(r["reference"]), normalize(r["hypothesis"])
    wer = jiwer.wer(rn, hn) if rn else float("nan")
    r.update(reference_norm=rn, hypothesis_norm=hn,
             wer_new=round(wer, 4), is_number_utt=int(is_number_utt(rn)))
    out.append(r)

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)

# ---- summary report ----
from collections import defaultdict
def mean(xs):
    xs = [x for x in xs if x == x]     # drop NaN
    return sum(xs) / len(xs) if xs else float("nan")

by = defaultdict(list)
for r in out:
    by[r["dataset"]].append(r)

print(f"\n{'dataset':<14} {'N':>3} {'WER(all)':>9} {'#num':>5} {'WER(no-num)':>12}")
print("-" * 48)
for ds, rs in by.items():
    allw  = [r["wer_new"] for r in rs]
    nonum = [r["wer_new"] for r in rs if not r["is_number_utt"]]
    print(f"{ds:<14} {len(rs):>3} {mean(allw)*100:>8.1f}% "
          f"{sum(r['is_number_utt'] for r in rs):>5} {mean(nonum)*100:>11.1f}%")

print("\nworst offenders after normalization:")
for r in sorted(out, key=lambda x: -(x['wer_new'] if x['wer_new'] == x['wer_new'] else -1))[:5]:
    print(f"  {r.get('sample_id','?'):<8} wer={r['wer_new']:.2f} num={r['is_number_utt']} "
          f"| ref='{r['reference_norm'][:40]}' hyp='{r['hypothesis_norm'][:40]}'")