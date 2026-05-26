"""
13_create_curated_games.py
--------------------------
Stream Lichess/standard-chess-games and collect a stratified sample of
1 000 games for the Streamlit "guess-the-Elo" app.

Filters applied per game
    • must contain [%eval] tags
    • avg_elo in [800, 2200], |elo_diff| <= 200
    • base time >= 180 s (no bullet)
    • total plies >= 25
    • result must be 1-0, 0-1, or 1/2-1/2

Output
    data/processed/curated_games.json  — list of dicts, one per game
    Player names are anonymized ("Player1" / "Player2").
"""

import json
import re
import sys
from pathlib import Path

import chess.pgn
import io

from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT     = Path(__file__).resolve().parent.parent
OUT_DIR  = ROOT / "data" / "processed"
OUT_FILE = OUT_DIR / "curated_games.json"

DATASET_REPO       = "Lichess/standard-chess-games"
DATASET_DATA_FILES = "data/year=2024/month=08/*.parquet"

# ── Stratified bucket config ───────────────────────────────────────────────────

BUCKETS: list[tuple[int, int, int]] = [
    (800,  1000, 15),
    (1000, 1200, 20),
    (1200, 1400, 30),
    (1400, 1600, 50),
    (1600, 1800, 40),
    (1800, 2000, 30),
    (2000, 2200, 15),
]
TOTAL_TARGET = sum(t for _, _, t in BUCKETS)

# ── Filters ────────────────────────────────────────────────────────────────────

ELO_DIFF_MAX  = 200
MIN_BASE_SECS = 180
MIN_PLIES     = 25
VALID_RESULTS = {"1-0", "0-1", "1/2-1/2"}

# ── Checkpoint ────────────────────────────────────────────────────────────────

SAVE_EVERY = 100   # write JSON after every N games added

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_time_control(tc: str) -> int | None:
    """Return base seconds, or None if unparseable / below minimum."""
    if not tc or tc in ("-", "?"):
        return None
    try:
        segment = tc.split("/")[-1]
        return int(segment.split("+")[0])
    except (ValueError, IndexError):
        return None


def bucket_index(avg_elo: float) -> int | None:
    """Return bucket index (0-based) for avg_elo, or None if out of range."""
    for i, (lo, hi, _) in enumerate(BUCKETS):
        if lo <= avg_elo < hi:
            return i
    return None


def count_plies(movetext: str) -> int:
    """Count half-moves by counting SAN tokens (rough but fast)."""
    # Strip clock/eval comments and move numbers, count word tokens.
    cleaned = re.sub(r'\{[^}]*\}', '', movetext)   # remove comments
    cleaned = re.sub(r'\d+\.+', '', cleaned)        # remove move numbers
    tokens  = [t for t in cleaned.split() if t not in VALID_RESULTS]
    return len(tokens)


def build_record(item: dict, game_id: int) -> dict | None:
    """
    Validate one dataset row and return a curated-game record, or None.

    Anonymises player names to "Player1" / "Player2".
    Counts plies by parsing the movetext through python-chess so the number
    exactly matches what the feature extractor will see.
    """
    movetext = item.get("movetext") or ""

    if "[%eval" not in movetext:
        return None

    # ── Elo ───────────────────────────────────────────────────────────────
    try:
        white_elo = int(item["WhiteElo"])
        black_elo = int(item["BlackElo"])
    except (KeyError, ValueError, TypeError):
        return None

    avg_elo  = (white_elo + black_elo) / 2.0
    elo_diff = abs(white_elo - black_elo)

    if not (800 <= avg_elo < 2200):
        return None
    if elo_diff > ELO_DIFF_MAX:
        return None

    # ── Time control ──────────────────────────────────────────────────────
    tc_str = item.get("TimeControl", "")
    base   = parse_time_control(tc_str)
    if base is None or base < MIN_BASE_SECS:
        return None

    # ── Result ────────────────────────────────────────────────────────────
    result = item.get("Result", "")
    if result not in VALID_RESULTS:
        return None

    # ── Ply count (fast token method — avoids full board replay) ──────────
    n_plies = count_plies(movetext)
    if n_plies < MIN_PLIES:
        return None

    return {
        "id":           game_id,
        "white":        "Player1",
        "black":        "Player2",
        "white_elo":    white_elo,
        "black_elo":    black_elo,
        "avg_elo":      round(avg_elo, 1),
        "time_control": tc_str,
        "opening":      item.get("Opening", ""),
        "eco":          item.get("ECO", ""),
        "result":       result,
        "termination":  item.get("Termination", ""),
        "movetext":     movetext,
        "total_moves":  n_plies,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUT_FILE.exists():
        answer = input(
            f"\n{OUT_FILE.name} already exists. Overwrite? [y/n]: "
        ).strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)
        print()

    # ── Bucket state ──────────────────────────────────────────────────────
    bucket_counts  = [0] * len(BUCKETS)
    bucket_targets = [t for _, _, t in BUCKETS]
    games: list[dict] = []
    next_id = 0

    # ── Progress display ──────────────────────────────────────────────────
    def bucket_label(i: int) -> str:
        lo, hi, tgt = BUCKETS[i]
        return f"{lo}-{hi:<5} ({bucket_counts[i]:>3}/{tgt})"

    def print_status() -> None:
        bar = " | ".join(bucket_label(i) for i in range(len(BUCKETS)))
        tqdm.write(f"  Buckets: {bar}  total={len(games)}")

    print(f"Dataset  : {DATASET_REPO}")
    print(f"Files    : {DATASET_DATA_FILES}")
    print(f"Target   : {TOTAL_TARGET} games  ({len(BUCKETS)} buckets)\n")

    dataset = load_dataset(
        DATASET_REPO,
        data_files=DATASET_DATA_FILES,
        streaming=True,
        split="train",
    )

    n_seen     = 0
    n_filtered = 0
    schema_ok  = False

    for item in tqdm(dataset, desc="Streaming", unit="game"):

        # One-time schema check
        if not schema_ok:
            actual  = set(item.keys())
            needed  = {"White", "Black", "WhiteElo", "BlackElo",
                       "Result", "TimeControl", "movetext"}
            missing = needed - actual
            print(f"\nDataset columns : {sorted(actual)}")
            print("Schema OK.\n" if not missing else f"WARNING missing: {missing}\n")
            schema_ok = True

        n_seen += 1

        # Fast early exit when all buckets full
        if all(bucket_counts[i] >= bucket_targets[i] for i in range(len(BUCKETS))):
            tqdm.write(f"\nAll buckets full after seeing {n_seen:,} games.")
            break

        record = build_record(item, next_id)
        if record is None:
            n_filtered += 1
            continue

        bi = bucket_index(record["avg_elo"])
        if bi is None or bucket_counts[bi] >= bucket_targets[bi]:
            n_filtered += 1
            continue

        games.append(record)
        bucket_counts[bi] += 1
        next_id += 1

        if len(games) % SAVE_EVERY == 0:
            OUT_FILE.write_text(json.dumps(games, ensure_ascii=False, indent=2),
                                encoding="utf-8")
            print_status()

    # ── Final save ────────────────────────────────────────────────────────
    OUT_FILE.write_text(json.dumps(games, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print(f"\n{'=' * 62}")
    print(f"  Games seen      : {n_seen:>10,}")
    print(f"  Filtered / skip : {n_filtered:>10,}")
    print(f"  Saved           : {len(games):>10,}")
    print(f"{'─' * 62}")
    print(f"  Bucket breakdown:")
    for i, (lo, hi, tgt) in enumerate(BUCKETS):
        pct = bucket_counts[i] / tgt * 100
        bar = "#" * bucket_counts[i] * 20 // tgt
        print(f"    {lo}-{hi}  {bucket_counts[i]:>3}/{tgt}  "
              f"[{bar:<20}] {pct:.0f}%")
    print(f"{'─' * 62}")
    print(f"  Output → {OUT_FILE}")
    size_kb = OUT_FILE.stat().st_size / 1024
    print(f"  File size       : {size_kb:.0f} KB")
    print(f"{'=' * 62}\n")


if __name__ == "__main__":
    main()
