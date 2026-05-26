"""
15_feature_extractor.py
-----------------------
Standalone feature extractor for a single chess PGN game.
Used by the Streamlit app to compute model features from any PGN string.

Entry point
-----------
    from python.feature_extractor import extract_features_from_pgn

    features: dict | None = extract_features_from_pgn(pgn_text)
    if features:
        df = pd.DataFrame([features])
        prediction = model.predict(df)[0]

Output schema (41 features, matches lichess_amateur_v2.parquet training data)
-------------------------------------------------------------------------------
  ACPL (8)         : white/black_acpl, _acpl_opening, _acpl_middle, _acpl_endgame
  Game meta (5)    : theory_depth_swing, total_moves,
                     time_control_seconds, time_control_increment, result_white
  Amateur (18)     : 9 patterns × 2 colors
  Time (10)        : 5 clock features × 2 colors

Dependencies: chess, chess.pgn — standard python-chess library only.
No pandas, no project imports, no model loading.
"""

import io
import re
import statistics
from typing import Optional

import chess
import chess.pgn


# ── Regex ──────────────────────────────────────────────────────────────────────

_EVAL_RE = re.compile(r'\[%eval\s+([^\]]+)\]')
_CLK_RE  = re.compile(r'\[%clk\s+([^\]]+)\]')

# ── ACPL constants ─────────────────────────────────────────────────────────────

_CAP_CP            = 1000   # centipawn cap for outlier moves
_SKIP_PLIES        = 8      # ignore first 8 plies (opening theory)
_OPEN_END          = 15     # opening/middlegame boundary (ply index)
_MID_END           = 40     # middlegame/endgame boundary (ply index)
_SWING_THRESHOLD   = 70     # cp threshold for theory-depth swing detection

# ── Pattern constants ──────────────────────────────────────────────────────────

_CORNER_SQUARES: dict[chess.Color, set[chess.Square]] = {
    chess.WHITE: {chess.A1, chess.H1},
    chess.BLACK: {chess.A8, chess.H8},
}

_A_H_FILES = frozenset({0, 7})   # file indices for a-file / h-file

# ── Output feature order (41 features, same as training parquet) ───────────────

FEATURE_COLUMNS = [
    # ACPL (8)
    "white_acpl",          "black_acpl",
    "white_acpl_opening",  "black_acpl_opening",
    "white_acpl_middle",   "black_acpl_middle",
    "white_acpl_endgame",  "black_acpl_endgame",
    # Game meta (5)
    "theory_depth_swing",
    "total_moves",
    "time_control_seconds", "time_control_increment",
    "result_white",
    # Amateur patterns (18 = 9 patterns × 2 colors)
    "white_pointless_checks",          "black_pointless_checks",
    "white_early_queen_blunders",      "black_early_queen_blunders",
    "white_opening_blunders",          "black_opening_blunders",
    "white_bad_corner_bishop",         "black_bad_corner_bishop",
    "white_bad_rim_knight",            "black_bad_rim_knight",
    "white_castled",                   "black_castled",
    "white_castled_move",              "black_castled_move",
    "white_pawn_moves_in_opening",     "black_pawn_moves_in_opening",
    "white_piece_moved_twice_opening", "black_piece_moved_twice_opening",
    # Time features (10 = 5 features × 2 colors)
    "white_avg_think",        "black_avg_think",
    "white_think_std",        "black_think_std",
    "white_fast_moves_pct",   "black_fast_moves_pct",
    "white_quick_blunders",   "black_quick_blunders",
    "white_opening_thinking", "black_opening_thinking",
]


# ══════════════════════════════════════════════════════════════════════════════
# Parse helpers
# ══════════════════════════════════════════════════════════════════════════════

def parse_eval_str(s: str) -> Optional[int]:
    """
    Convert a Lichess eval string to centipawns (White's POV).

    Examples
    --------
    '1.23'  → 123
    '-0.45' → -45
    '#3'    → 1000  (mate-in-3 for the side to move)
    '#-2'   → -1000 (opponent mates in 2)
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


def parse_time_control(tc: str) -> tuple[Optional[int], Optional[int]]:
    """
    Parse a PGN TimeControl header into (base_seconds, increment_seconds).

    Handles formats: "300+5", "600", "40/9000+30", "-", "?".
    Returns (None, None) for unrecognised strings.
    """
    if not tc or tc in ("-", "?", ""):
        return None, None
    try:
        segment = tc.split("/")[-1]
        parts   = segment.split("+")
        base    = int(parts[0])
        inc     = int(parts[1]) if len(parts) > 1 else 0
        return base, inc
    except (ValueError, IndexError):
        return None, None


def parse_clk_str(s: str) -> Optional[int]:
    """
    Parse a Lichess clock comment string to total seconds.

    Accepts 'H:MM:SS' and 'M:SS' formats.

    Examples
    --------
    '0:05:00' → 300
    '1:23:45' → 5025
    '0:00:45' → 45
    """
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


# ══════════════════════════════════════════════════════════════════════════════
# Game-level extraction helpers
# ══════════════════════════════════════════════════════════════════════════════

def extract_evals(game: chess.pgn.Game) -> list[Optional[int]]:
    """
    Return one centipawn evaluation per ply (0-indexed), White's POV.

    Values come from Lichess [%eval ...] comment annotations.
    Plies without an annotation are represented as None.
    """
    evals: list[Optional[int]] = []
    node = game
    while node.variations:
        node = node.variations[0]
        m = _EVAL_RE.search(node.comment)
        evals.append(parse_eval_str(m.group(1)) if m else None)
    return evals


def extract_clocks(game: chess.pgn.Game) -> list[Optional[int]]:
    """
    Return one clock reading per ply (0-indexed), seconds remaining after move.

    Values come from Lichess [%clk ...] comment annotations.
    Plies without an annotation are represented as None.
    """
    clocks: list[Optional[int]] = []
    node = game
    while node.variations:
        node = node.variations[0]
        m = _CLK_RE.search(node.comment)
        clocks.append(parse_clk_str(m.group(1)) if m else None)
    return clocks


# ══════════════════════════════════════════════════════════════════════════════
# ACPL computation
# ══════════════════════════════════════════════════════════════════════════════

def acpl_range(
    evals:     list[Optional[int]],
    color:     chess.Color,
    idx_start: int,
    idx_end:   int,
) -> Optional[float]:
    """
    Average Centipawn Loss for `color` over ply indices [idx_start, idx_end).

    Loss per move = max(0, eval_before − eval_after) from the mover's POV.
    Capped at _CAP_CP to reduce outlier influence.
    Returns None if no eligible plies exist.
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
        losses.append(max(0.0, min(float(_CAP_CP), float(raw))))
    return round(statistics.mean(losses), 2) if losses else None


def theory_depth_swing(
    evals:        list[Optional[int]],
    threshold_cp: int = _SWING_THRESHOLD,
) -> Optional[int]:
    """
    Return the ply number (1-indexed) at which the evaluation first drifts
    more than `threshold_cp` centipawns from the game's opening position.

    A proxy for how long the opening 'theory' phase lasts.
    Returns None if evals are unavailable or no swing is detected.
    """
    if not evals or evals[0] is None:
        return None
    baseline = evals[0]
    for idx in range(1, len(evals)):
        if evals[idx] is None:
            continue
        if abs(evals[idx] - baseline) >= threshold_cp:
            return idx + 1
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Amateur pattern extraction  (board replay)
# ══════════════════════════════════════════════════════════════════════════════

def extract_amateur_patterns(
    game:  chess.pgn.Game,
    evals: list[Optional[int]],
) -> tuple[dict[chess.Color, dict], int]:
    """
    Replay the game move-by-move and detect nine amateur-pattern features
    for each color.

    Returns
    -------
    (patterns, total_moves)
        patterns : dict keyed by chess.WHITE / chess.BLACK, each mapping
                   feature name → int value
        total_moves : number of half-moves (plies) in the game

    Patterns detected
    -----------------
    1. pointless_checks       — gave check but evaluation dropped ≥ 200 cp
    2. early_queen_blunders   — queen moved before move 5 with ≥ 150 cp drop
    3. opening_blunders       — any move in first 15 plies with ≥ 300 cp drop
    4. bad_corner_bishop      — bishop retreated to own back corner (≥ 100 cp drop)
    5. bad_rim_knight         — knight moved to a/h-file in first 20 plies (≥ 100 cp)
    6. castled                — 1 if player castled, else 0
    7. castled_move           — fullmove number when castled (−1 if never)
    8. pawn_moves_in_opening  — pawn moves in first 15 own plies
    9. piece_moved_twice_opening — non-pawn pieces moved ≥ 2× in first 15 own plies
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

    own_ply_count: dict[chess.Color, int]          = {chess.WHITE: 0, chess.BLACK: 0}
    piece_move_history: dict[chess.Color, dict[int, int]] = {
        chess.WHITE: {},
        chess.BLACK: {},
    }
    total_moves = 0

    for ply_idx, move in enumerate(game.mainline_moves()):
        total_moves += 1
        mover    = board.turn
        piece    = board.piece_at(move.from_square)
        fullmove = board.fullmove_number

        in_own_opening = own_ply_count[mover] < 15

        # Eval drop from mover's POV
        e_before = evals[ply_idx - 1] if ply_idx > 0            else None
        e_after  = evals[ply_idx]     if ply_idx < len(evals)   else None
        if e_before is not None and e_after is not None:
            drop: Optional[float] = float(
                (e_before - e_after) if mover == chess.WHITE
                else (e_after - e_before)
            )
        else:
            drop = None

        is_castling = board.is_castling(move)
        board.push(move)
        gives_check = board.is_check()

        if piece is not None:
            pt = piece.piece_type
            p  = patterns[mover]

            # 1. Pointless check
            if gives_check and drop is not None and drop >= 200:
                p["pointless_checks"] += 1

            # 2. Early queen blunder
            if pt == chess.QUEEN and fullmove < 10 and drop is not None and drop >= 150:
                p["early_queen_blunders"] += 1

            # 3. Opening blunder (first 15 total plies)
            if ply_idx < 15 and drop is not None and drop >= 300:
                p["opening_blunders"] += 1

            # 4. Bad corner bishop
            if (pt == chess.BISHOP
                    and move.to_square in _CORNER_SQUARES[mover]
                    and drop is not None and drop >= 100):
                p["bad_corner_bishop"] += 1

            # 5. Bad rim knight (first 20 total plies)
            if (pt == chess.KNIGHT
                    and ply_idx < 20
                    and chess.square_file(move.to_square) in _A_H_FILES
                    and drop is not None and drop >= 100):
                p["bad_rim_knight"] += 1

            # 6 & 7. Castling
            if is_castling and p["castled"] == 0:
                p["castled"]      = 1
                p["castled_move"] = fullmove

            # 8. Pawn moves in own opening
            if in_own_opening and pt == chess.PAWN:
                p["pawn_moves_in_opening"] += 1

            # 9. Non-pawn piece moved twice in own opening
            if pt != chess.PAWN:
                from_sq  = move.from_square
                to_sq    = move.to_square
                prev_cnt = piece_move_history[mover].get(from_sq, 0)
                new_cnt  = prev_cnt + 1
                piece_move_history[mover].pop(from_sq, None)
                piece_move_history[mover][to_sq] = new_cnt
                if in_own_opening and new_cnt == 2:
                    p["piece_moved_twice_opening"] += 1

        own_ply_count[mover] += 1

    return patterns, total_moves


# ══════════════════════════════════════════════════════════════════════════════
# Thinking-time features
# ══════════════════════════════════════════════════════════════════════════════

def compute_thinking_times(
    clocks:    list[Optional[int]],
    base:      int,
    increment: int,
) -> list[Optional[float]]:
    """
    Convert per-ply remaining-clock readings to per-ply thinking times.

    Lichess semantics: clocks[i] is time remaining *after* move i.
    The previous clock for the same color is clocks[i−2] (or base_time
    for the first move of each color).

    Formula: thinking_time[i] = prev_clock_same_color − clocks[i] + increment

    Negative values are clamped to 0 to absorb Lichess rounding artefacts.
    Plies without clock data produce None.
    """
    times: list[Optional[float]] = []
    for i, clk in enumerate(clocks):
        if clk is None:
            times.append(None)
            continue
        prev_clk: Optional[int] = clocks[i - 2] if i >= 2 else base
        if prev_clk is None:
            times.append(None)
            continue
        times.append(max(0.0, float(prev_clk - clk + increment)))
    return times


def extract_time_features(
    thinking_times: list[Optional[float]],
    color:          chess.Color,
    evals:          list[Optional[int]],
) -> dict:
    """
    Compute five clock-based features for one color.

    Features
    --------
    avg_think        : mean thinking time per move (seconds)
    think_std        : standard deviation of thinking times
    fast_moves_pct   : fraction of moves made in < 3 seconds
    quick_blunders   : moves with thinking_time < 5 s AND eval drop ≥ 200 cp
    opening_thinking : total thinking time in the first 15 plies (both colors)

    All values are None when no clock data is available for this color.
    quick_blunders and opening_thinking are always int/float (defaulting to 0).
    """
    own_times:        list[float] = []
    opening_thinking: float       = 0.0
    quick_blunders:   int         = 0
    n_plies = len(thinking_times)

    for ply_idx in range(n_plies):
        # Filter to this color's moves only
        if (ply_idx % 2 == 0) != (color == chess.WHITE):
            continue

        t = thinking_times[ply_idx]

        if ply_idx < 15 and t is not None:
            opening_thinking += t

        if t is None:
            continue

        own_times.append(t)

        if t < 5:
            e_before = evals[ply_idx - 1] if ply_idx > 0          else None
            e_after  = evals[ply_idx]     if ply_idx < len(evals)  else None
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


# ══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════════════════

def extract_features_from_pgn(pgn_text: str) -> Optional[dict]:
    """
    Extract all 41 model features from a single PGN string.

    The PGN is expected to contain Lichess-style comment annotations:
      [%eval X.XX] — engine evaluation in pawns after each move
      [%clk H:MM:SS] — clock remaining after each move (optional)

    Parameters
    ----------
    pgn_text : str
        Complete PGN text, including headers and movetext.

    Returns
    -------
    dict
        Mapping of feature_name → value, with keys matching FEATURE_COLUMNS.
        Time features will be None if [%clk] tags are absent.
    None
        If the PGN cannot be parsed, has no eval annotations,
        the game is too short (< 10 plies), or WhiteElo/BlackElo headers
        are missing.
    """
    try:
        game = chess.pgn.read_game(io.StringIO(pgn_text))
    except Exception:
        return None

    if game is None:
        return None

    headers = game.headers

    # ── Elo validation ─────────────────────────────────────────────────────
    try:
        white_elo = int(headers.get("WhiteElo", ""))
        black_elo = int(headers.get("BlackElo", ""))
    except (ValueError, TypeError):
        return None

    # ── Time control ───────────────────────────────────────────────────────
    base, inc = parse_time_control(headers.get("TimeControl", ""))
    base = base if base is not None else 0
    inc  = inc  if inc  is not None else 0

    # ── Result ─────────────────────────────────────────────────────────────
    result_map  = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}
    result_str  = headers.get("Result", "*")
    result_white = result_map.get(result_str)     # None for unfinished/unknown

    # ── Evals (required) ──────────────────────────────────────────────────
    evals = extract_evals(game)
    if not evals or all(e is None for e in evals):
        return None

    n = len(evals)

    # ── Clocks (optional) ─────────────────────────────────────────────────
    clocks = extract_clocks(game)

    # ── ACPL features ─────────────────────────────────────────────────────
    w_acpl  = acpl_range(evals, chess.WHITE, _SKIP_PLIES, n)
    b_acpl  = acpl_range(evals, chess.BLACK, _SKIP_PLIES, n)
    w_open  = acpl_range(evals, chess.WHITE, 0,           _OPEN_END)
    b_open  = acpl_range(evals, chess.BLACK, 0,           _OPEN_END)
    w_mid   = acpl_range(evals, chess.WHITE, _OPEN_END,   _MID_END)
    b_mid   = acpl_range(evals, chess.BLACK, _OPEN_END,   _MID_END)
    w_end   = acpl_range(evals, chess.WHITE, _MID_END,    n)
    b_end   = acpl_range(evals, chess.BLACK, _MID_END,    n)
    swing   = theory_depth_swing(evals)

    # ── Amateur patterns + total_moves ────────────────────────────────────
    patterns, total_moves = extract_amateur_patterns(game, evals)

    if total_moves < 10:
        return None

    wp = patterns[chess.WHITE]
    bp = patterns[chess.BLACK]

    # ── Thinking-time features ────────────────────────────────────────────
    thinking_times = compute_thinking_times(clocks, base, inc)
    wt = extract_time_features(thinking_times, chess.WHITE, evals)
    bt = extract_time_features(thinking_times, chess.BLACK, evals)

    # ── Assemble output dict (key order matches FEATURE_COLUMNS) ──────────
    return {
        # ACPL
        "white_acpl":             w_acpl,
        "black_acpl":             b_acpl,
        "white_acpl_opening":     w_open,
        "black_acpl_opening":     b_open,
        "white_acpl_middle":      w_mid,
        "black_acpl_middle":      b_mid,
        "white_acpl_endgame":     w_end,
        "black_acpl_endgame":     b_end,
        # Game meta
        "theory_depth_swing":     swing,
        "total_moves":            total_moves,
        "time_control_seconds":   base,
        "time_control_increment": inc,
        "result_white":           result_white,
        # Amateur patterns
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


# ══════════════════════════════════════════════════════════════════════════════
# Quick smoke-test
# ══════════════════════════════════════════════════════════════════════════════

_SAMPLE_PGN = """\
[Event "Rated Rapid game"]
[Site "https://lichess.org/test"]
[Date "2024.06.01"]
[White "PlayerA"]
[Black "PlayerB"]
[WhiteElo "1500"]
[BlackElo "1480"]
[Result "1-0"]
[TimeControl "600+0"]

1. e4 { [%eval 0.17] [%clk 0:10:00] } 1... c5 { [%eval 0.19] [%clk 0:10:00] }
2. Nf3 { [%eval 0.25] [%clk 0:09:55] } 2... d6 { [%eval 0.22] [%clk 0:09:58] }
3. d4 { [%eval 0.35] [%clk 0:09:50] } 3... cxd4 { [%eval 0.31] [%clk 0:09:55] }
4. Nxd4 { [%eval 0.28] [%clk 0:09:47] } 4... Nf6 { [%eval 0.30] [%clk 0:09:52] }
5. Nc3 { [%eval 0.32] [%clk 0:09:44] } 5... a6 { [%eval 0.28] [%clk 0:09:49] }
6. Bg5 { [%eval 0.45] [%clk 0:09:40] } 6... e6 { [%eval 0.40] [%clk 0:09:46] }
7. f4 { [%eval 0.55] [%clk 0:09:37] } 7... Qb6 { [%eval 0.48] [%clk 0:09:43] }
8. Qd2 { [%eval 0.50] [%clk 0:09:34] } 8... Qxb2 { [%eval 1.20] [%clk 0:09:40] }
9. Rb1 { [%eval 1.10] [%clk 0:09:30] } 9... Qa3 { [%eval 0.90] [%clk 0:09:36] }
10. e5 { [%eval 1.40] [%clk 0:09:26] } 10... dxe5 { [%eval 1.20] [%clk 0:09:32] }
11. fxe5 { [%eval 1.35] [%clk 0:09:22] } 11... Nfd7 { [%eval 1.15] [%clk 0:09:28] }
12. Ne4 { [%eval 1.50] [%clk 0:09:18] } 12... Bb4 { [%eval 1.30] [%clk 0:09:24] }
1-0
"""


if __name__ == "__main__":
    print("Running smoke test on sample PGN...\n")

    features = extract_features_from_pgn(_SAMPLE_PGN)

    if features is None:
        print("ERROR: extract_features_from_pgn returned None")
    else:
        print(f"Feature count : {len(features)}  (expected {len(FEATURE_COLUMNS)})")
        print(f"Keys match    : {set(features) == set(FEATURE_COLUMNS)}\n")

        # Pretty-print grouped
        groups = [
            ("ACPL",           [k for k in FEATURE_COLUMNS if "acpl" in k or k == "theory_depth_swing"]),
            ("Game meta",      ["total_moves", "time_control_seconds", "time_control_increment", "result_white"]),
            ("Amateur",        [k for k in FEATURE_COLUMNS if any(p in k for p in [
                "checks", "queen_blunder", "opening_blunder", "corner", "rim",
                "castled", "pawn_moves", "piece_moved"])]),
            ("Time",           [k for k in FEATURE_COLUMNS if any(p in k for p in [
                "avg_think", "think_std", "fast_moves", "quick_blunder", "opening_thinking"])]),
        ]
        for group_name, keys in groups:
            print(f"  ── {group_name}")
            for k in keys:
                v = features.get(k)
                print(f"    {k:<36}  {v}")
            print()

        missing = [c for c in FEATURE_COLUMNS if c not in features]
        if missing:
            print(f"WARNING: missing columns: {missing}")
        else:
            print("All expected columns present. Smoke test passed.")
