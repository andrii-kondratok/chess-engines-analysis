"""
08_etl_lichess.py
-----------------
Stream Lichess/standard-chess-games (HuggingFace) for month 2024-06.
Filter games with [%eval] annotations, extract ACPL and game-phase
features from existing Lichess analysis tags (no Stockfish needed).
Collects separately:
  - up to MAX_RAPID    rapid    games (600 < base_secs <= 1800)
  - up to MAX_CLASSICAL classical games (base_secs > 1800)
Saves each type to its own Parquet file.
 
Dataset rows are structured dicts (not raw PGN text). Headers are read
directly from dict fields; movetext is parsed by python-chess for evals.
 
ACPL definition (from eval tags, White's POV throughout):
    White loss at ply i = max(0, eval[i-1] - eval[i])   ← eval dropped
    Black loss at ply i = max(0, eval[i]   - eval[i-1]) ← eval rose
Both capped at CAP_CP. First SKIP_PLIES plies excluded from overall ACPL.
Mate scores (#N) → ±1000 cp.
"""
 
import io
import re
import statistics
from pathlib import Path
 
import chess
import chess.pgn
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()
 
 
# ── Config ─────────────────────────────────────────────────────────────────────
 
ROOT     = Path(__file__).resolve().parent.parent
OUT_DIR  = ROOT / "data" / "processed"
OUT_RAPID     = OUT_DIR / "lichess_rapid.parquet"
OUT_CLASSICAL = OUT_DIR / "lichess_classical.parquet"
 
DATASET_REPO       = "Lichess/standard-chess-games"
DATASET_DATA_FILES = "data/year=2024/month=07/*.parquet"
 
# Expected columns — verified against first row at startup
EXPECTED_COLS = {"White", "Black", "WhiteElo", "BlackElo",
                 "Result", "TimeControl", "movetext"}
 
MAX_RAPID     = 100_000   # rapid:    600 < base_secs <= 1800
MAX_CLASSICAL =   2_000   # classical: base_secs > 1800
SAVE_EVERY    =  100
 
# ── TEST MODE ──────────────────────────────────────────────────────────────────
# Stop after seeing this many games that have [%eval] tags (before other filters).
# Set to None to disable and run the full pipeline.
# TO REMOVE: delete the three lines below and the "if TEST_EVAL_SEEN" block in main().
TEST_EVAL_SEEN: int | None = None
PILOT_PGN = ROOT / "data" / "raw" / "lichess_pilot_evals.pgn"
 
# ── Filters ────────────────────────────────────────────────────────────────────
 
AVG_ELO_MIN    = 800
AVG_ELO_MAX    = 2200
ELO_DIFF_MAX   = 200
MIN_MOVES      = 10     # total half-moves (plies)
 
# Time control boundaries (base seconds)
RAPID_MIN      = 600    # > 600 s  (exclusive lower bound to avoid blitz edge)
RAPID_MAX      = 1800   # ≤ 1800 s  (30-min games included)
CLASSICAL_MIN  = 1801   # > 1800 s
 
# ── ACPL params ────────────────────────────────────────────────────────────────
 
CAP_CP      = 1000
SKIP_PLIES  = 8        # exclude from overall ACPL (opening theory)
 
# Phase boundaries: half-move (ply) indices, 0-based
# opening → [0,  15)  plies 1-15
# middle  → [15, 40)  plies 16-40
# endgame → [40, ∞)   plies 41+
OPEN_END  = 15
MID_END   = 40
 
SWING_THRESHOLD_CP = 70   # 0.7 pawns
 
EVAL_RE = re.compile(r'\[%eval\s+([^\]]+)\]')
 
COLUMNS = [
    "avg_elo", "elo_diff",
    "white_acpl", "black_acpl",
    "white_acpl_opening", "black_acpl_opening",
    "white_acpl_middle",  "black_acpl_middle",
    "white_acpl_endgame", "black_acpl_endgame",
    "theory_depth_swing",
    "total_moves",
    "time_control_seconds", "time_control_increment",
    "result_white",
]
 
 
# ── Parse helpers ───────────────────────────────────────────────────────────────
 
def parse_eval_str(s: str) -> int | None:
    """
    Parse the inner value of a [%eval ...] tag to centipawns (White's POV).
    Handles floats ("0.25" → 25 cp) and mate scores ("#5" → 1000, "#-3" → -1000).
    Returns None on parse failure.
    """
    s = s.strip()
    if s.startswith('#'):
        try:
            n = int(s[1:])
            return 1000 if n > 0 else -1000
        except ValueError:
            return None
    try:
        return int(float(s) * 100)
    except ValueError:
        return None
 
 
def parse_time_control(tc: str) -> tuple[int | None, int | None]:
    """
    Parse a PGN TimeControl string → (base_seconds, increment_seconds).
    Handles "300+3", "600", "40/9000+30". Returns (None, None) on failure.
    """
    if not tc or tc in ("-", "?"):
        return None, None
    try:
        segment = tc.split("/")[-1]   # strip "40/" prefix if present
        parts   = segment.split("+")
        base    = int(parts[0])
        inc     = int(parts[1]) if len(parts) > 1 else 0
        return base, inc
    except (ValueError, IndexError):
        return None, None
 
 
def classify_tc(base: int) -> str | None:
    """Return 'rapid', 'classical', or None (skip)."""
    if RAPID_MIN < base <= RAPID_MAX:
        return "rapid"
    if base >= CLASSICAL_MIN:
        return "classical"
    return None   # bullet / blitz → skip
 

def result_to_float(result: str) -> float | None:
    return {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get(result)
 
 
# ── Eval extraction ─────────────────────────────────────────────────────────────
 
def extract_evals(game: chess.pgn.Game) -> list[int | None]:
    """
    Walk mainline; return one eval per ply (0-indexed).
    evals[i] = centipawns from White's POV immediately after ply i was played.
    White plays on even ply indices (0, 2, 4, …), Black on odd (1, 3, 5, …).
    """
    evals: list[int | None] = []
    node = game
    while node.variations:
        node = node.variations[0]
        m    = EVAL_RE.search(node.comment)
        evals.append(parse_eval_str(m.group(1)) if m else None)
    return evals
 
 
# ── ACPL computation ────────────────────────────────────────────────────────────
 
def acpl_range(
    evals: list[int | None],
    color: chess.Color,
    idx_start: int,
    idx_end: int,
) -> float | None:
    """
    Compute ACPL for `color` over half-move (ply) indices [idx_start, idx_end).
 
    For each ply `idx` in range:
      - mover = WHITE if idx is even, BLACK if idx is odd
      - skip if mover != color
      - White loss = max(0, evals[idx-1] - evals[idx])   (eval dropped)
      - Black loss = max(0, evals[idx]   - evals[idx-1]) (eval rose from White's POV)
      - cap at CAP_CP
 
    Returns mean loss in centipawns, or None if no valid moves in range.
    """
    losses: list[float] = []
 
    for idx in range(max(idx_start, 1), min(idx_end, len(evals))):
        mover = chess.WHITE if idx % 2 == 0 else chess.BLACK
        if mover != color:
            continue
 
        prev = evals[idx - 1]
        curr = evals[idx]
        if prev is None or curr is None:
            continue
 
        raw = (prev - curr) if color == chess.WHITE else (curr - prev)
        losses.append(max(0.0, min(float(CAP_CP), float(raw))))
 
    return round(statistics.mean(losses), 2) if losses else None
 
 
def theory_depth_swing(
    evals: list[int | None],
    threshold_cp: int = SWING_THRESHOLD_CP,
) -> int | None:
    """
    Return the 1-indexed ply number where the eval first deviates
    >= threshold_cp from the baseline (eval after ply 0 = after White's first move).
    Returns None if baseline is missing or the threshold is never reached.
    """
    if not evals or evals[0] is None:
        return None
    baseline = evals[0]
    for idx in range(1, len(evals)):
        if evals[idx] is None:
            continue
        if abs(evals[idx] - baseline) >= threshold_cp:
            return idx + 1   # convert to 1-indexed ply number
    return None
 
 
# ── Sanity-check helper ─────────────────────────────────────────────────────────
 
def reconstruct_pgn(item: dict) -> str:
    """Rebuild a standard PGN string from structured dataset columns."""
    headers = []
    for tag in ("Event", "Site", "Result", "White", "Black",
                "WhiteElo", "BlackElo", "ECO", "Opening",
                "TimeControl", "Termination", "UTCDate", "UTCTime"):
        val = item.get(tag)
        if val is not None and val != "":
            headers.append(f'[{tag} "{val}"]')
    return "\n".join(headers) + "\n\n" + (item.get("movetext") or "") + "\n\n"
 
 
# ── Feature extraction ──────────────────────────────────────────────────────────
 
def extract_features(item: dict) -> dict | None:
    """
    Process one structured dataset row.
    Headers are read directly from the dict (no PGN header parsing needed).
    A minimal PGN string is built from movetext for python-chess eval walking.
    Returns a feature dict (keys match COLUMNS) or None if filtered/invalid.
    All exceptions are caught — never crashes on malformed input.
    """
    try:
        movetext = item.get("movetext") or ""
 
        # Fast pre-filter: skip unannotated games before any parsing
        if '[%eval' not in movetext:
            return None
 
        # ── Elo — already int in this dataset ────────────────────────────
        try:
            white_elo = int(item["WhiteElo"])
            black_elo = int(item["BlackElo"])
        except (KeyError, ValueError, TypeError):
            return None
 
        avg_elo  = (white_elo + black_elo) / 2.0
        elo_diff = abs(white_elo - black_elo)
 
        if not (AVG_ELO_MIN <= avg_elo <= AVG_ELO_MAX):
            return None
        if elo_diff > ELO_DIFF_MAX:
            return None
 
        # ── Time control ──────────────────────────────────────────────────
        base, inc = parse_time_control(item.get("TimeControl", ""))
        if base is None:
            return None
        tc_type = classify_tc(base)
        if tc_type is None:
            return None   # bullet / blitz → skip
 
        # ── Result ────────────────────────────────────────────────────────
        result_str = item.get("Result", "")
        result_w   = result_to_float(result_str)
        if result_w is None:
            return None
 
        # ── Build minimal PGN and parse with python-chess ─────────────────
        # Only the Result header is needed; python-chess uses it for game end.
        pgn_text = f'[Result "{result_str}"]\n\n{movetext}'
        game     = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None:
            return None
 
        # ── Move count ────────────────────────────────────────────────────
        total_moves = sum(1 for _ in game.mainline_moves())
        if total_moves < MIN_MOVES:
            return None
 
        # ── Evals ─────────────────────────────────────────────────────────
        evals = extract_evals(game)
        if not evals or all(e is None for e in evals):
            return None
 
        n = len(evals)
 
        # Overall ACPL — skip first SKIP_PLIES plies (opening theory)
        w_acpl = acpl_range(evals, chess.WHITE, SKIP_PLIES, n)
        b_acpl = acpl_range(evals, chess.BLACK, SKIP_PLIES, n)
 
        # Phase ACPLs — no skip, use full phase range
        w_open = acpl_range(evals, chess.WHITE, 0,        OPEN_END)
        b_open = acpl_range(evals, chess.BLACK, 0,        OPEN_END)
        w_mid  = acpl_range(evals, chess.WHITE, OPEN_END, MID_END)
        b_mid  = acpl_range(evals, chess.BLACK, OPEN_END, MID_END)
        w_end  = acpl_range(evals, chess.WHITE, MID_END,  n)
        b_end  = acpl_range(evals, chess.BLACK, MID_END,  n)
 
        swing = theory_depth_swing(evals)
 
        return {
            "avg_elo":                round(avg_elo, 1),
            "elo_diff":               elo_diff,
            "white_acpl":             w_acpl,
            "black_acpl":             b_acpl,
            "white_acpl_opening":     w_open,
            "black_acpl_opening":     b_open,
            "white_acpl_middle":      w_mid,
            "black_acpl_middle":      b_mid,
            "white_acpl_endgame":     w_end,
            "black_acpl_endgame":     b_end,
            "theory_depth_swing":     swing,
            "total_moves":            total_moves,
            "time_control_seconds":   base,
            "time_control_increment": inc,
            "result_white":           result_w,
            "_tc_type":               tc_type,   # internal routing key, stripped before save
        }
 
    except Exception:
        return None   # never crash on malformed input
 
 
# ── Main ────────────────────────────────────────────────────────────────────────
 
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
 
    print(f"Dataset  : {DATASET_REPO}")
    print(f"Files    : {DATASET_DATA_FILES}")
    print(f"Targets  : rapid={MAX_RAPID:,}  |  classical={MAX_CLASSICAL:,}")
    print(f"Filters  : avg_elo [{AVG_ELO_MIN},{AVG_ELO_MAX}]  "
          f"|  elo_diff ≤ {ELO_DIFF_MAX}  "
          f"|  plies ≥ {MIN_MOVES}\n")
 
    dataset = load_dataset(
        DATASET_REPO,
        data_files=DATASET_DATA_FILES,
        streaming=True,
        split="train",
    )
 
    rapid_rows:     list[dict] = []
    classical_rows: list[dict] = []
 
    n_seen      = 0
    n_no_eval   = 0
    n_filtered  = 0
    schema_verified = False
 
    pilot_pgn_fh = open(PILOT_PGN, "a", encoding="utf-8") if TEST_EVAL_SEEN is not None else None
 
    def _save_checkpoint(label: str, rows: list[dict], path: Path) -> None:
        pd.DataFrame(
            [{k: v for k, v in r.items() if k != "_tc_type"} for r in rows],
            columns=COLUMNS,
        ).to_parquet(path, index=False)
        tqdm.write(f"  [checkpoint] {label}={len(rows):,}  seen={n_seen:,}")
 
    for item in tqdm(dataset, desc="Streaming", unit="game"):
        # Both quotas filled → stop
        if len(rapid_rows) >= MAX_RAPID and len(classical_rows) >= MAX_CLASSICAL:
            tqdm.write("\nBoth quotas reached. Stopping.")
            break
 
        # ── Schema check ──────────────────────────────────────────────────
        if not schema_verified:
            actual_cols = set(item.keys())
            missing     = EXPECTED_COLS - actual_cols
            print(f"\nDataset columns : {sorted(actual_cols)}")
            if missing:
                print(f"WARNING — missing expected columns: {missing}")
            else:
                print("Schema OK — all expected columns present.")
            print()
            schema_verified = True
 
        movetext = item.get("movetext") or ""
        n_seen  += 1
 
        if '[%eval' not in movetext:
            n_no_eval += 1
            continue
 
        if pilot_pgn_fh is not None:
            pilot_pgn_fh.write(reconstruct_pgn(item))
            pilot_pgn_fh.flush()
 
        if TEST_EVAL_SEEN is not None and (n_seen - n_no_eval) >= TEST_EVAL_SEEN:
            tqdm.write(f"\n[TEST MODE] Reached {TEST_EVAL_SEEN} eval-tagged games. Stopping.")
            break
 
        features = extract_features(item)
        if features is None:
            n_filtered += 1
            continue
 
        tc_type = features["_tc_type"]
 
        if tc_type == "rapid" and len(rapid_rows) < MAX_RAPID:
            rapid_rows.append(features)
            if len(rapid_rows) % SAVE_EVERY == 0:
                _save_checkpoint("rapid", rapid_rows, OUT_RAPID)
 
        elif tc_type == "classical" and len(classical_rows) < MAX_CLASSICAL:
            classical_rows.append(features)
            if len(classical_rows) % SAVE_EVERY == 0:
                _save_checkpoint("classical", classical_rows, OUT_CLASSICAL)
 
        else:
            n_filtered += 1   # quota already full for this tc_type
 
    if pilot_pgn_fh is not None:
        pilot_pgn_fh.close()
 
    # ── Final saves ────────────────────────────────────────────────────────
    def _save_final(rows: list[dict], path: Path, label: str) -> pd.DataFrame:
        df = pd.DataFrame(
            [{k: v for k, v in r.items() if k != "_tc_type"} for r in rows],
            columns=COLUMNS,
        )
        df.to_parquet(path, index=False)
        print(f"  {label}: {len(df):,} rows → {path}")
        return df
 
    print(f"\n{'═' * 56}")
    print(f"  Games seen      : {n_seen:>10,}")
    print(f"  No eval tags    : {n_no_eval:>10,}  ({n_no_eval / max(n_seen, 1) * 100:5.1f}%)")
    print(f"  Filtered out    : {n_filtered:>10,}  ({n_filtered / max(n_seen, 1) * 100:5.1f}%)")
    print(f"{'─' * 56}")
 
    df_rapid     = _save_final(rapid_rows,     OUT_RAPID,     "Rapid    ")
    df_classical = _save_final(classical_rows, OUT_CLASSICAL, "Classical")
 
    print(f"{'═' * 56}\n")
    for label, df in [("RAPID", df_rapid), ("CLASSICAL", df_classical)]:
        print(f"\n── {label} ──")
        print(df.describe(percentiles=[.25, .5, .75]).to_string())
 
 
if __name__ == "__main__":
    main()
 