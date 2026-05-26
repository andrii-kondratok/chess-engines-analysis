"""
11_etl_amateur_patterns.py
--------------------------
Re-streams Lichess/standard-chess-games (2024-06) and extracts:
  • Existing 15 base features (ACPL, time-control, result)
  • 18 amateur board-pattern features (9 per color, via board replay)
  • 10 thinking-time features (5 per color, parsed from [%clk] tags)

Total output: 43 columns.

Thinking-time semantics (Lichess):
  clock[i] = time REMAINING after move i (ply-indexed, 0-based)
  prev_clock_same_color[i] = clock[i-2] if i >= 2, else base_time
  thinking_time[i] = prev_clock_same_color[i] - clock[i] + increment

Games without [%clk] tags produce NaN for all 10 time features.
"""

import io
import re
import statistics
import sys
from pathlib import Path

import chess
import chess.pgn
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


# ── Config ─────────────────────────────────────────────────────────────────────

ROOT     = Path(__file__).resolve().parent.parent
OUT_DIR  = ROOT / "data" / "processed"
OUT_FILE = OUT_DIR / "lichess_amateur_v2.parquet"

DATASET_REPO       = "Lichess/standard-chess-games"
DATASET_DATA_FILES = "data/year=2024/month=06/*.parquet"

EXPECTED_COLS = {"White", "Black", "WhiteElo", "BlackElo",
                 "Result", "TimeControl", "movetext"}

MAX_GAMES_KEPT = 600_000
SAVE_EVERY     =   100

# ── Filters (unchanged from script 08) ────────────────────────────────────────

AVG_ELO_MIN   = 800
AVG_ELO_MAX   = 2200
ELO_DIFF_MAX  = 200
MIN_BASE_SECS = 180
MIN_MOVES     = 10

# ── ACPL params (unchanged) ────────────────────────────────────────────────────

CAP_CP     = 1000
SKIP_PLIES = 8
OPEN_END   = 15
MID_END    = 40

SWING_THRESHOLD_CP = 70

EVAL_RE = re.compile(r'\[%eval\s+([^\]]+)\]')
CLK_RE  = re.compile(r'\[%clk\s+([^\]]+)\]')

# ── Pattern constants ──────────────────────────────────────────────────────────

# Corners associated with each color's trapped bishops
CORNER_SQUARES = {
    chess.WHITE: {chess.A1, chess.H1},
    chess.BLACK: {chess.A8, chess.H8},
}

A_H_FILES = frozenset({0, 7})   # file indices for a-file and h-file

# ── Output columns ─────────────────────────────────────────────────────────────

BASE_COLUMNS = [
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

PATTERN_NAMES = [
    "pointless_checks",
    "early_queen_blunders",
    "opening_blunders",
    "bad_corner_bishop",
    "bad_rim_knight",
    "castled",
    "castled_move",
    "pawn_moves_in_opening",
    "piece_moved_twice_opening",
]

PATTERN_COLUMNS = [
    f"{color}_{pat}"
    for pat in PATTERN_NAMES
    for color in ("white", "black")
]

TIME_FEATURE_NAMES = [
    "avg_think",
    "think_std",
    "fast_moves_pct",
    "quick_blunders",
    "opening_thinking",
]

TIME_COLUMNS = [
    f"{color}_{feat}"
    for feat in TIME_FEATURE_NAMES
    for color in ("white", "black")
]

COLUMNS = BASE_COLUMNS + PATTERN_COLUMNS + TIME_COLUMNS


# ── Parse helpers (identical to script 08) ─────────────────────────────────────

def parse_eval_str(s: str) -> int | None:
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
    if not tc or tc in ("-", "?"):
        return None, None
    try:
        segment = tc.split("/")[-1]
        parts   = segment.split("+")
        base    = int(parts[0])
        inc     = int(parts[1]) if len(parts) > 1 else 0
        return base, inc
    except (ValueError, IndexError):
        return None, None


def result_to_float(result: str) -> float | None:
    return {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get(result)


def parse_clock(s: str) -> int | None:
    """Parse 'H:MM:SS' (or 'M:SS') clock string → total seconds."""
    s = s.strip()
    try:
        parts = s.split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
    except (ValueError, IndexError):
        pass
    return None


# ── Eval extraction (identical to script 08) ───────────────────────────────────

def extract_evals(game: chess.pgn.Game) -> list[int | None]:
    """One eval per ply (0-indexed), centipawns from White's POV."""
    evals: list[int | None] = []
    node = game
    while node.variations:
        node = node.variations[0]
        m    = EVAL_RE.search(node.comment)
        evals.append(parse_eval_str(m.group(1)) if m else None)
    return evals


# ── Clock extraction ───────────────────────────────────────────────────────────

def extract_clocks(game: chess.pgn.Game) -> list[int | None]:
    """One clock reading per ply (0-indexed), seconds remaining after the move."""
    clocks: list[int | None] = []
    node = game
    while node.variations:
        node = node.variations[0]
        m    = CLK_RE.search(node.comment)
        clocks.append(parse_clock(m.group(1)) if m else None)
    return clocks


def compute_thinking_times(
    clocks:    list[int | None],
    base:      int,
    increment: int,
) -> list[float | None]:
    """
    Convert per-ply remaining-clock values to per-ply thinking times.

    Lichess semantics: clocks[i] is time remaining AFTER move i.
    The previous clock for the same color is clocks[i-2] (i >= 2) or base_time.

    thinking_time[i] = prev_clock_same_color - clocks[i] + increment
    Clamped to >= 0 to absorb rounding artefacts.
    """
    times: list[float | None] = []
    for i, clk in enumerate(clocks):
        if clk is None:
            times.append(None)
            continue
        prev_clk: int | None = clocks[i - 2] if i >= 2 else base
        if prev_clk is None:
            times.append(None)
            continue
        times.append(max(0.0, float(prev_clk - clk + increment)))
    return times


def time_features_for_color(
    thinking_times: list[float | None],
    color:          chess.Color,
    evals:          list[int | None],
) -> dict:
    """
    Compute 5 thinking-time features for one color.

    Returns a dict with keys:
        avg_think, think_std, fast_moves_pct, quick_blunders, opening_thinking
    All values are float | None (NaN-safe).
    """
    own_times:        list[float] = []
    opening_thinking: float       = 0.0
    quick_blunders:   int         = 0
    n_plies = len(thinking_times)

    for ply_idx in range(n_plies):
        if (ply_idx % 2 == 0) != (color == chess.WHITE):
            continue                      # not this color's move

        t = thinking_times[ply_idx]

        # Opening thinking time: first 15 total plies
        if ply_idx < 15 and t is not None:
            opening_thinking += t

        if t is None:
            continue

        own_times.append(t)

        # Quick blunders: fast move + large eval drop
        if t < 5:
            e_before = evals[ply_idx - 1] if ply_idx > 0 else None
            e_after  = evals[ply_idx]     if ply_idx < len(evals) else None
            if e_before is not None and e_after is not None:
                drop = (e_before - e_after) if color == chess.WHITE \
                       else (e_after - e_before)
                if drop >= 200:
                    quick_blunders += 1

    if not own_times:
        return {
            "avg_think":        None,
            "think_std":        None,
            "fast_moves_pct":   None,
            "quick_blunders":   quick_blunders,
            "opening_thinking": round(opening_thinking, 2),
        }

    avg_think      = statistics.mean(own_times)
    think_std      = statistics.stdev(own_times) if len(own_times) >= 2 else None
    fast_moves_pct = sum(1 for t in own_times if t < 3) / len(own_times)

    return {
        "avg_think":        round(avg_think, 2),
        "think_std":        round(think_std, 2) if think_std is not None else None,
        "fast_moves_pct":   round(fast_moves_pct, 4),
        "quick_blunders":   quick_blunders,
        "opening_thinking": round(opening_thinking, 2),
    }


# ── ACPL computation (identical to script 08) ──────────────────────────────────

def acpl_range(
    evals: list[int | None],
    color: chess.Color,
    idx_start: int,
    idx_end: int,
) -> float | None:
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
    if not evals or evals[0] is None:
        return None
    baseline = evals[0]
    for idx in range(1, len(evals)):
        if evals[idx] is None:
            continue
        if abs(evals[idx] - baseline) >= threshold_cp:
            return idx + 1
    return None


# ── Amateur pattern extraction ─────────────────────────────────────────────────

def extract_amateur_patterns(
    game: chess.pgn.Game,
    evals: list[int | None],
) -> tuple[dict, int]:
    """
    Replay the game, extracting amateur patterns alongside a move count.
    Returns (patterns_dict, total_moves).

    patterns_dict keys: chess.WHITE and chess.BLACK, each mapping
    pattern name → int value.

    Eval-aware patterns use:
        eval_before = evals[ply_idx - 1]  (eval AFTER previous ply, White's POV)
        eval_after  = evals[ply_idx]       (eval AFTER this ply, White's POV)
        drop (mover's POV) = eval_before - eval_after  for White
                           = eval_after  - eval_before  for Black

    Simple patterns use board state only (no eval required).
    """
    board = game.board()

    patterns: dict[chess.Color, dict] = {
        color: {
            "pointless_checks":          0,
            "early_queen_blunders":      0,
            "opening_blunders":          0,
            "bad_corner_bishop":         0,
            "bad_rim_knight":            0,
            "castled":                   0,
            "castled_move":             -1,
            "pawn_moves_in_opening":     0,
            "piece_moved_twice_opening": 0,
        }
        for color in (chess.WHITE, chess.BLACK)
    }

    # own_ply_count[color] = number of moves that color has made so far
    own_ply_count: dict[chess.Color, int] = {chess.WHITE: 0, chess.BLACK: 0}

    # piece_move_history[color][current_square] = times that piece has been moved
    # Updated every time a non-pawn piece moves, regardless of phase.
    piece_move_history: dict[chess.Color, dict[int, int]] = {
        chess.WHITE: {},
        chess.BLACK: {},
    }

    total_moves = 0

    for ply_idx, move in enumerate(game.mainline_moves()):
        total_moves += 1
        mover   = board.turn
        piece   = board.piece_at(move.from_square)
        fullmove = board.fullmove_number

        # Phase flags (based on own-move count before this move)
        in_own_opening = own_ply_count[mover] < 15   # first 15 own moves
        in_own_early20 = own_ply_count[mover] < 20   # first 20 own moves

        # ── Eval drop from mover's POV ─────────────────────────────────────
        eval_before = evals[ply_idx - 1] if ply_idx > 0 else None
        eval_after  = evals[ply_idx]     if ply_idx < len(evals) else None

        if eval_before is not None and eval_after is not None:
            drop: float | None = float(
                (eval_before - eval_after) if mover == chess.WHITE
                else (eval_after - eval_before)
            )
        else:
            drop = None

        # ── Pre-push checks ────────────────────────────────────────────────
        is_castling = board.is_castling(move)

        # ── Advance board ──────────────────────────────────────────────────
        board.push(move)
        gives_check = board.is_check()

        # ── Pattern detection ──────────────────────────────────────────────
        if piece is not None:
            pt = piece.piece_type
            p  = patterns[mover]

            # 1. Pointless check: gave check but lost eval
            if gives_check and drop is not None and drop >= 200:
                p["pointless_checks"] += 1

            # 2. Early queen blunder: queen out before move 5 with big drop
            if pt == chess.QUEEN and fullmove < 5 and drop is not None:
                p["early_queen_blunders"] += 1

            # 3. Opening blunder: any move in first 15 TOTAL plies with huge drop
            if ply_idx < 15 and drop is not None and drop >= 300:
                p["opening_blunders"] += 1

            # 4. Bad corner bishop: bishop parked in own back corner
            if (pt == chess.BISHOP
                    and move.to_square in CORNER_SQUARES[mover]
                    and drop is not None and drop >= 100):
                p["bad_corner_bishop"] += 1

            # 5. Bad rim knight: knight to a/h-file in first 20 TOTAL plies
            if (pt == chess.KNIGHT
                    and ply_idx < 20
                    and chess.square_file(move.to_square) in A_H_FILES
                    and drop is not None and drop >= 100):
                p["bad_rim_knight"] += 1

            # 6 & 7. Castling (first occurrence only)
            if is_castling and p["castled"] == 0:
                p["castled"]      = 1
                p["castled_move"] = fullmove

            # 8. Pawn moves in own opening
            if in_own_opening and pt == chess.PAWN:
                p["pawn_moves_in_opening"] += 1

            # 9. Non-pawn piece moved twice within own opening
            #    Always update piece_move_history (piece square changes after move).
            if pt != chess.PAWN:
                from_sq   = move.from_square
                to_sq     = move.to_square
                prev_cnt  = piece_move_history[mover].get(from_sq, 0)
                new_cnt   = prev_cnt + 1
                piece_move_history[mover].pop(from_sq, None)
                piece_move_history[mover][to_sq] = new_cnt
                if in_own_opening and new_cnt == 2:
                    p["piece_moved_twice_opening"] += 1

        own_ply_count[mover] += 1

    return patterns, total_moves


# ── Feature extraction (combines 08 logic + patterns) ─────────────────────────

def extract_features(item: dict) -> dict | None:
    """
    Returns a feature dict with all BASE_COLUMNS + PATTERN_COLUMNS,
    or None if the game fails any filter. Never raises.
    """
    try:
        movetext = item.get("movetext") or ""

        if '[%eval' not in movetext:
            return None

        # ── Elo (already int in this dataset) ─────────────────────────────
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

        # ── Time control ───────────────────────────────────────────────────
        base, inc = parse_time_control(item.get("TimeControl", ""))
        if base is None or base < MIN_BASE_SECS:
            return None

        # ── Result ─────────────────────────────────────────────────────────
        result_str = item.get("Result", "")
        result_w   = result_to_float(result_str)
        if result_w is None:
            return None

        # ── Parse game ─────────────────────────────────────────────────────
        pgn_text = f'[Result "{result_str}"]\n\n{movetext}'
        game     = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None:
            return None

        # ── Evals ──────────────────────────────────────────────────────────
        evals = extract_evals(game)
        if not evals or all(e is None for e in evals):
            return None

        n = len(evals)

        # ── ACPL features ──────────────────────────────────────────────────
        w_acpl = acpl_range(evals, chess.WHITE, SKIP_PLIES, n)
        b_acpl = acpl_range(evals, chess.BLACK, SKIP_PLIES, n)
        w_open = acpl_range(evals, chess.WHITE, 0,        OPEN_END)
        b_open = acpl_range(evals, chess.BLACK, 0,        OPEN_END)
        w_mid  = acpl_range(evals, chess.WHITE, OPEN_END, MID_END)
        b_mid  = acpl_range(evals, chess.BLACK, OPEN_END, MID_END)
        w_end  = acpl_range(evals, chess.WHITE, MID_END,  n)
        b_end  = acpl_range(evals, chess.BLACK, MID_END,  n)
        swing  = theory_depth_swing(evals)

        # ── Amateur patterns (board replay — also returns total_moves) ──────
        patterns, total_moves = extract_amateur_patterns(game, evals)

        if total_moves < MIN_MOVES:
            return None

        # ── Thinking-time features ──────────────────────────────────────────
        clocks          = extract_clocks(game)
        thinking_times  = compute_thinking_times(clocks, base, inc)
        wt = time_features_for_color(thinking_times, chess.WHITE, evals)
        bt = time_features_for_color(thinking_times, chess.BLACK, evals)

        # ── Flatten pattern dict ────────────────────────────────────────────
        wp = patterns[chess.WHITE]
        bp = patterns[chess.BLACK]

        return {
            # Base features
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
            # Pattern features
            "white_pointless_checks":           wp["pointless_checks"],
            "black_pointless_checks":           bp["pointless_checks"],
            "white_early_queen_blunders":       wp["early_queen_blunders"],
            "black_early_queen_blunders":       bp["early_queen_blunders"],
            "white_opening_blunders":           wp["opening_blunders"],
            "black_opening_blunders":           bp["opening_blunders"],
            "white_bad_corner_bishop":          wp["bad_corner_bishop"],
            "black_bad_corner_bishop":          bp["bad_corner_bishop"],
            "white_bad_rim_knight":             wp["bad_rim_knight"],
            "black_bad_rim_knight":             bp["bad_rim_knight"],
            "white_castled":                    wp["castled"],
            "black_castled":                    bp["castled"],
            "white_castled_move":               wp["castled_move"],
            "black_castled_move":               bp["castled_move"],
            "white_pawn_moves_in_opening":      wp["pawn_moves_in_opening"],
            "black_pawn_moves_in_opening":      bp["pawn_moves_in_opening"],
            "white_piece_moved_twice_opening":  wp["piece_moved_twice_opening"],
            "black_piece_moved_twice_opening":  bp["piece_moved_twice_opening"],
            # Time features
            "white_avg_think":        wt["avg_think"],
            "black_avg_think":        bt["avg_think"],
            "white_think_std":        wt["think_std"],
            "black_think_std":        bt["think_std"],
            "white_fast_moves_pct":   wt["fast_moves_pct"],
            "black_fast_moves_pct":   bt["fast_moves_pct"],
            "white_quick_blunders":   wt["quick_blunders"],
            "black_quick_blunders":   bt["quick_blunders"],
            "white_opening_thinking": wt["opening_thinking"],
            "black_opening_thinking": bt["opening_thinking"],
        }

    except Exception:
        return None


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Handle existing output ─────────────────────────────────────────────
    if OUT_FILE.exists():
        answer = input(
            f"\n{OUT_FILE.name} already exists. "
            f"Overwrite? [y / bak / n]: "
        ).strip().lower()
        if answer == "bak":
            backup = OUT_FILE.with_suffix(".bak.parquet")
            OUT_FILE.rename(backup)
            print(f"  Renamed existing file → {backup.name}\n")
        elif answer == "y":
            OUT_FILE.unlink()
            print()
        else:
            print("Aborted.")
            sys.exit(0)

    print(f"Dataset  : {DATASET_REPO}")
    print(f"Files    : {DATASET_DATA_FILES}")
    print(f"Max keep : {MAX_GAMES_KEPT:,}   |   Save every : {SAVE_EVERY:,}")
    print(f"Output   : {OUT_FILE.name}  ({len(COLUMNS)} columns)")
    print(f"Filters  : avg_elo [{AVG_ELO_MIN},{AVG_ELO_MAX}]  "
          f"|  elo_diff ≤ {ELO_DIFF_MAX}  "
          f"|  base ≥ {MIN_BASE_SECS}s  "
          f"|  plies ≥ {MIN_MOVES}\n")

    dataset = load_dataset(
        DATASET_REPO,
        data_files=DATASET_DATA_FILES,
        streaming=True,
        split="train",
    )

    rows:           list[dict] = []
    n_seen          = 0
    n_no_eval       = 0
    n_filtered      = 0
    n_kept          = 0
    schema_verified = False

    for item in tqdm(dataset, desc="Streaming", unit="game"):

        # ── Schema check (first row only) ──────────────────────────────────
        if not schema_verified:
            actual = set(item.keys())
            missing = EXPECTED_COLS - actual
            print(f"\nDataset columns : {sorted(actual)}")
            print("Schema OK." if not missing
                  else f"WARNING — missing: {missing}")
            print()
            schema_verified = True

        movetext = item.get("movetext") or ""
        n_seen  += 1

        if '[%eval' not in movetext:
            n_no_eval += 1
            continue

        features = extract_features(item)
        if features is None:
            n_filtered += 1
            continue

        rows.append(features)
        n_kept += 1

        if n_kept % SAVE_EVERY == 0:
            pd.DataFrame(rows, columns=COLUMNS).to_parquet(OUT_FILE, index=False)
            tqdm.write(f"  [checkpoint] kept={n_kept:,}  seen={n_seen:,}  "
                       f"rate={n_kept / n_seen * 100:.1f}%")

        if n_kept >= MAX_GAMES_KEPT:
            tqdm.write(f"\nReached MAX_GAMES_KEPT={MAX_GAMES_KEPT:,}. Stopping.")
            break

    # ── Final save ─────────────────────────────────────────────────────────
    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_parquet(OUT_FILE, index=False)

    print(f"\n{'═' * 58}")
    print(f"  Games seen      : {n_seen:>10,}")
    print(f"  No eval tags    : {n_no_eval:>10,}  ({n_no_eval / max(n_seen, 1) * 100:5.1f}%)")
    print(f"  Filtered out    : {n_filtered:>10,}  ({n_filtered / max(n_seen, 1) * 100:5.1f}%)")
    print(f"  Kept            : {n_kept:>10,}  ({n_kept    / max(n_seen, 1) * 100:5.1f}%)")
    print(f"{'─' * 58}")
    print(f"  Output → {OUT_FILE}")
    print(f"{'═' * 58}\n")

    # ── Pattern feature summary ────────────────────────────────────────────
    print("Pattern feature means (White / Black):")
    for pat in PATTERN_NAMES:
        wc = f"white_{pat}"
        bc = f"black_{pat}"
        wm = df[wc].mean() if wc in df.columns else float("nan")
        bm = df[bc].mean() if bc in df.columns else float("nan")
        print(f"  {pat:<32}  W={wm:.3f}  B={bm:.3f}")

    # ── Time feature summary ───────────────────────────────────────────────
    clk_coverage = df["white_avg_think"].notna().sum()
    print(f"\nTime feature means (White / Black)  "
          f"[clock coverage: {clk_coverage:,} / {len(df):,} rows "
          f"= {clk_coverage / max(len(df), 1) * 100:.1f}%]:")
    for feat in TIME_FEATURE_NAMES:
        wc = f"white_{feat}"
        bc = f"black_{feat}"
        wm = df[wc].mean() if wc in df.columns else float("nan")
        bm = df[bc].mean() if bc in df.columns else float("nan")
        print(f"  {feat:<32}  W={wm:.3f}  B={bm:.3f}")


if __name__ == "__main__":
    main()
