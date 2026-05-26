from datasets import load_dataset
from collections import defaultdict

ds = load_dataset(
    "Lichess/standard-chess-games",
    data_files="data/year=2024/month=07/*.parquet",
    streaming=True, split="train"
)

n_total = 0
n_eval = 0
tc_counts = defaultdict(int)

for item in ds:
    n_total += 1
    movetext = item.get("movetext") or ""
    
    if '[%eval' in movetext:
        n_eval += 1
        tc = item.get("TimeControl", "?")
        try:
            base = int(tc.split("+")[0].split("/")[-1])
            if base < 180:       bucket = "bullet"
            elif base < 600:     bucket = "blitz"
            elif base <= 1800:   bucket = "rapid"
            else:                bucket = "classical"
        except:
            bucket = "unknown"
        tc_counts[bucket] += 1

    if n_total % 10_000 == 0:
        print(f"\n── {n_total:,} ігор ──")
        print(f"  з eval: {n_eval:,} ({n_eval/n_total*100:.1f}%)")
        for k, v in sorted(tc_counts.items()):
            print(f"  {k}: {v:,}")

    if n_total >= 200_000:
        break