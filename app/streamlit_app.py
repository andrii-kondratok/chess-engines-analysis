"""
app/streamlit_app.py
--------------------
"Guess the Elo" — Streamlit app where users see an anonymised chess game
and compete against the ML model to estimate the players' average Elo.

Run locally:
    cd app
    streamlit run streamlit_app.py
"""

import json
import random
import re
from io import StringIO
from pathlib import Path

import chess
import chess.pgn
import chess.svg
import joblib
import pandas as pd
import streamlit as st

from feature_extractor import FEATURE_COLUMNS, extract_features_from_pgn

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Guess the Elo",
    page_icon="♟️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Paths (relative to this file — works both locally and on HF Spaces) ───────

APP_DIR    = Path(__file__).parent
GAMES_FILE = APP_DIR / "curated_games.json"
MODEL_FILE = APP_DIR / "elo_predictor_v3_1.pkl"

# ── Material helpers ───────────────────────────────────────────────────────────

PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9,
}

# Black Unicode symbols (shown as pieces white captured from black)
_SYM_BLACK = {
    chess.PAWN: "♟", chess.KNIGHT: "♞", chess.BISHOP: "♝",
    chess.ROOK: "♜", chess.QUEEN: "♛",
}
# White Unicode symbols (shown as pieces black captured from white)
_SYM_WHITE = {
    chess.PAWN: "♙", chess.KNIGHT: "♘", chess.BISHOP: "♗",
    chess.ROOK: "♖", chess.QUEEN: "♕",
}

_PIECE_ORDER = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
_STARTING    = {chess.PAWN: 8, chess.KNIGHT: 2, chess.BISHOP: 2,
                chess.ROOK: 2, chess.QUEEN: 1}


def compute_material(board: chess.Board) -> tuple[int, str, str]:
    """
    Returns (white_advantage, white_captured_str, black_captured_str).

    white_advantage    : positive = white ahead, negative = black ahead.
    white_captured_str : black pieces white has taken (shown on WHITE row, bottom).
    black_captured_str : white pieces black has taken (shown on BLACK row, top).
    """
    white_cnt = {pt: 0 for pt in _STARTING}
    black_cnt = {pt: 0 for pt in _STARTING}
    for piece in board.piece_map().values():
        if piece.piece_type in _STARTING:
            if piece.color == chess.WHITE:
                white_cnt[piece.piece_type] += 1
            else:
                black_cnt[piece.piece_type] += 1

    w_captured: list[chess.PieceType] = []   # black pieces taken by white
    b_captured: list[chess.PieceType] = []   # white pieces taken by black
    for pt, start in _STARTING.items():
        w_captured.extend([pt] * (start - black_cnt[pt]))
        b_captured.extend([pt] * (start - white_cnt[pt]))

    w_captured.sort(key=lambda p: _PIECE_ORDER.index(p))
    b_captured.sort(key=lambda p: _PIECE_ORDER.index(p))

    score_w = sum(PIECE_VALUES[pt] * white_cnt[pt] for pt in _STARTING)
    score_b = sum(PIECE_VALUES[pt] * black_cnt[pt] for pt in _STARTING)

    return (
        score_w - score_b,
        "".join(_SYM_BLACK[p] for p in w_captured),
        "".join(_SYM_WHITE[p] for p in b_captured),
    )


# ── Clock helpers ──────────────────────────────────────────────────────────────

_CLK_RE = re.compile(r'\[%clk\s+(\d+):(\d+):(\d+)\]')


def extract_clocks_from_movetext(movetext: str) -> list[int | None]:
    """Walk PGN nodes and return clock remaining (seconds) after each ply."""
    game = chess.pgn.read_game(StringIO(f'[Result "*"]\n\n{movetext}'))
    if game is None:
        return []
    clocks: list[int | None] = []
    node = game
    while node.variations:
        node = node.variations[0]
        m = _CLK_RE.search(node.comment)
        if m:
            h, mn, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
            clocks.append(h * 3600 + mn * 60 + s)
        else:
            clocks.append(None)
    return clocks


@st.cache_data
def extract_clocks_for_game(game_idx: int) -> tuple[list[int | None], list[int | None]]:
    """
    Cached per game_idx.
    Returns (white_clocks, black_clocks) — one entry per move of that colour.
    Index 0 = after move 1, index 1 = after move 2, etc.
    """
    games      = load_games()
    all_clocks = extract_clocks_from_movetext(games[game_idx]["movetext"])
    # ply 0 (index 0) = white's 1st move; ply 1 = black's 1st move; ...
    white_clocks = [all_clocks[i] for i in range(0, len(all_clocks), 2)]
    black_clocks = [all_clocks[i] for i in range(1, len(all_clocks), 2)]
    return white_clocks, black_clocks


def get_clocks_at_ply(
    clocks_tuple: tuple[list, list], current_ply: int
) -> tuple[int | None, int | None]:
    """
    Return (white_clock, black_clock) reflecting the most recent tick
    for each colour up to and including current_ply (1-indexed).
    """
    white_clocks, black_clocks = clocks_tuple
    n_white = (current_ply + 1) // 2   # white moves made so far
    n_black = current_ply // 2         # black moves made so far
    w = white_clocks[n_white - 1] if 0 < n_white <= len(white_clocks) else None
    b = black_clocks[n_black - 1] if 0 < n_black <= len(black_clocks) else None
    return w, b


def format_clock(seconds: int | None) -> str:
    if seconds is None:
        return "--:--"
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


# ── Cached loaders ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    """Load LightGBM model once; pin in memory for the session."""
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_games() -> list[dict]:
    """Load curated games JSON once."""
    with open(GAMES_FILE, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def get_features(game_idx: int) -> dict | None:
    """
    Extract features for one game (board replay, eval parsing).
    Cached per game_idx so repeated reruns never redo the work.
    """
    games   = load_games()
    game    = games[game_idx]
    pgn_str = reconstruct_pgn(game)
    return extract_features_from_pgn(pgn_str)


# ── PGN helpers ────────────────────────────────────────────────────────────────

def reconstruct_pgn(game: dict) -> str:
    """
    Build a minimal PGN string from a curated game dict.
    WhiteElo/BlackElo are included so feature_extractor can read them,
    but the UI never shows them before the guess is submitted.
    """
    header_lines = [
        '[Event "Lichess game"]',
        '[Site "https://lichess.org"]',
        f'[White "{game["white"]}"]',
        f'[Black "{game["black"]}"]',
        f'[WhiteElo "{game["white_elo"]}"]',
        f'[BlackElo "{game["black_elo"]}"]',
        f'[Result "{game["result"]}"]',
        f'[TimeControl "{game["time_control"]}"]',
        f'[ECO "{game.get("eco") or "?"}"]',
        f'[Opening "{game.get("opening") or "?"}"]',
        f'[Termination "{game.get("termination") or "?"}"]',
    ]
    return "\n".join(header_lines) + "\n\n" + game["movetext"] + "\n"


def parse_moves(pgn_str: str) -> list[chess.Move]:
    """Return mainline move list from a PGN string."""
    parsed = chess.pgn.read_game(StringIO(pgn_str))
    return list(parsed.mainline_moves()) if parsed else []


def board_at_ply(moves: list[chess.Move], ply: int) -> chess.Board:
    """Return board state after `ply` half-moves from the start."""
    board = chess.Board()
    for move in moves[:ply]:
        board.push(move)
    return board


# ── Prediction ─────────────────────────────────────────────────────────────────

def compute_prediction(game_idx: int) -> float | None:
    """
    Predict avg_elo for the game at game_idx.
    Feature extraction is cached; prediction itself is fast.
    """
    features = get_features(game_idx)
    if features is None:
        return None
    model = load_model()
    feature_order = list(model.feature_name_)
    X = pd.DataFrame(
        [{col: features.get(col) for col in feature_order}],
        dtype=float,
    )
    try:
        return float(model.predict(X)[0])
    except Exception:
        return None


# ── Session state ──────────────────────────────────────────────────────────────

def _init_state(n_games: int) -> None:
    defaults: dict = {
        "game_idx":   random.randint(0, n_games - 1),
        "submitted":  False,
        "counted":    False,
        "user_guess": 1500,
        "wins_user":  0,
        "wins_model": 0,
        "ties":       0,
        "move_idx":   1,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _new_game(n_games: int) -> None:
    """Pick a new random game (different from current), reset round state."""
    current = st.session_state.game_idx
    idx = current
    while idx == current and n_games > 1:
        idx = random.randint(0, n_games - 1)
    st.session_state.game_idx  = idx
    st.session_state.submitted = False
    st.session_state.counted   = False
    st.session_state.move_idx  = 1
    st.rerun()


# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar(n_games: int) -> None:
    st.sidebar.title("♟️ Guess the Elo")
    st.sidebar.caption(f"{n_games} games loaded")
    st.sidebar.markdown("---")

    st.sidebar.markdown("### Session score")
    c1, c2, c3 = st.sidebar.columns(3)
    c1.metric("You 🧠",    st.session_state.wins_user)
    c2.metric("Model 🤖", st.session_state.wins_model)
    c3.metric("Ties 🤝",  st.session_state.ties)

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 New game", use_container_width=True):
        _new_game(n_games)

    with st.sidebar.expander("📖 How to play"):
        st.markdown(
            """
            1. Step through moves with ⏮ ◀ ▶ ⏭.
            2. Estimate the **average Elo** of both players.
            3. Click **Submit ✅** and see how you compare to the ML model!

            **Winning:** whoever's guess is closer to the real Elo wins the
            round. Ties when errors are equal.

            **Elo scale:** 800 Beginner · 1200 Club · 1500 Intermediate
            · 1800 Advanced · 2000+ Expert

            **About the model:** it uses centipawn loss (accuracy),
            opening patterns (blunders, queen sorties, castling) and
            per-move thinking time extracted from Lichess clock tags.
            """
        )


# ── Feature breakdown helper ───────────────────────────────────────────────────

def _render_feature_breakdown(feats: dict, actual_elo: float) -> None:
    """Show a human-readable summary of key features for the current game."""
    w_acpl  = feats.get("white_acpl")
    b_acpl  = feats.get("black_acpl")
    w_think = feats.get("white_avg_think")
    b_think = feats.get("black_avg_think")

    rows = []
    if w_acpl is not None:
        rows.append(("White ACPL",      f"{w_acpl:.1f} cp",  "Lower = more accurate"))
    if b_acpl is not None:
        rows.append(("Black ACPL",      f"{b_acpl:.1f} cp",  ""))
    if w_think is not None:
        rows.append(("White avg think", f"{w_think:.1f} s",  "Time per move"))
    if b_think is not None:
        rows.append(("Black avg think", f"{b_think:.1f} s",  ""))

    rows += [
        ("White castled",      str(bool(feats.get("white_castled"))),         ""),
        ("Black castled",      str(bool(feats.get("black_castled"))),         ""),
        ("Opening blunders W", str(feats.get("white_opening_blunders", 0)),  "≥300 cp drop in first 15 plies"),
        ("Opening blunders B", str(feats.get("black_opening_blunders", 0)),  ""),
        ("Theory depth swing", str(feats.get("theory_depth_swing")),         "Ply where eval leaves opening"),
    ]

    df = pd.DataFrame(rows, columns=["Feature", "Value", "Note"])
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(
        f"**Actual Elo:** {actual_elo:.0f}  ·  "
        f"ACPL reference: ~800 Elo ≈ 80 cp, ~1500 ≈ 35 cp, ~2000 ≈ 15 cp"
    )


# ── Board panel helper ─────────────────────────────────────────────────────────

def _player_row_html(
    captured_str: str,
    adv: int,
    clock_str: str,
    active: bool,
) -> str:
    """
    One HTML row: captured pieces + material advantage on the left,
    clock on the right. active=True highlights the clock in blue/bold.
    """
    adv_html   = f"<span style='color:#28a745;font-size:0.9em'>&nbsp;+{adv}</span>" if adv > 0 else ""
    clk_color  = "#2E86AB" if active else "#888"
    clk_weight = "bold"    if active else "normal"
    return (
        f"<div style='display:flex;justify-content:space-between;"
        f"align-items:center;padding:3px 6px;font-size:1.05em'>"
        f"<span style='font-size:1.25em'>{captured_str}</span>{adv_html}"
        f"<span style='font-family:monospace;color:{clk_color};"
        f"font-weight:{clk_weight};font-size:1.15em'>{clock_str}</span>"
        f"</div>"
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    games   = load_games()
    n_games = len(games)
    _init_state(n_games)
    render_sidebar(n_games)

    game_idx = st.session_state.game_idx
    game     = games[game_idx]

    # Kick off feature extraction + prediction early (result is cached).
    predicted_elo = compute_prediction(game_idx)

    # ── Game header (full width) ──────────────────────────────────────────
    opening = game.get("opening") or "?"
    eco     = game.get("eco") or ""
    st.markdown(
        f"## {opening}"
        + (f"  <span style='color:grey;font-size:0.8em'>({eco})</span>" if eco else ""),
        unsafe_allow_html=True,
    )
    st.markdown(
        f"**Time control:** `{game['time_control']}`  &nbsp;|&nbsp;  "
        f"**Result:** `{game['result']}`  &nbsp;|&nbsp;  "
        f"**Plies:** {game['total_moves']}"
    )

    # ── Parse moves ───────────────────────────────────────────────────────
    pgn_str     = reconstruct_pgn(game)
    moves       = parse_moves(pgn_str)
    total_plies = len(moves)

    if total_plies == 0:
        st.error("Could not parse moves for this game.")
        if st.button("Skip to next game"):
            _new_game(n_games)
        return

    # Clamp stored move_idx to valid range for this game
    stored_idx = max(1, min(st.session_state.move_idx, total_plies))
    fullmove   = (stored_idx + 1) // 2
    mover      = "White" if stored_idx % 2 == 1 else "Black"

    # ── Two-column layout: board | controls ──────────────────────────────
    col_board, col_controls = st.columns([3, 2])

    # ── LEFT: chess board with material + clocks ──────────────────────────
    with col_board:
        board     = board_at_ply(moves, stored_idx)
        last_move = moves[stored_idx - 1] if stored_idx > 0 else None

        advantage, white_captured, black_captured = compute_material(board)

        clocks_tuple              = extract_clocks_for_game(game_idx)
        white_clock, black_clock  = get_clocks_at_ply(clocks_tuple, stored_idx)

        active_color = "white" if board.turn == chess.WHITE else "black"

        # Black is at the top of the board
        black_adv = max(0, -advantage)
        st.markdown(
            _player_row_html(
                black_captured,         # pieces BLACK captured (white symbols ♙♘♗♖♕)
                black_adv,
                format_clock(black_clock),
                active=(active_color == "black"),
            ),
            unsafe_allow_html=True,
        )

        svg_str = chess.svg.board(
            board,
            size=380,
            lastmove=last_move,
            colors={
                "square light":          "#f0d9b5",
                "square dark":           "#b58863",
                "square light lastmove": "#cdd16e",
                "square dark lastmove":  "#aaa23a",
            },
        )
        st.markdown(
            f"<div style='display:flex;justify-content:center;margin:2px 0'>"
            f"{svg_str}</div>",
            unsafe_allow_html=True,
        )

        # White is at the bottom of the board
        white_adv = max(0, advantage)
        st.markdown(
            _player_row_html(
                white_captured,         # pieces WHITE captured (black symbols ♟♞♝♜♛)
                white_adv,
                format_clock(white_clock),
                active=(active_color == "white"),
            ),
            unsafe_allow_html=True,
        )

    # ── RIGHT: navigation + guess/results ────────────────────────────────
    with col_controls:

        # 4-button navigation row
        nav_cols = st.columns(4)
        with nav_cols[0]:
            if st.button("⏮", help="Go to start",
                         use_container_width=True, disabled=(stored_idx <= 1)):
                st.session_state.move_idx = 1
                st.rerun()
        with nav_cols[1]:
            if st.button("◀", help="Previous move",
                         use_container_width=True, disabled=(stored_idx <= 1)):
                st.session_state.move_idx = stored_idx - 1
                st.rerun()
        with nav_cols[2]:
            if st.button("▶", help="Next move",
                         use_container_width=True, disabled=(stored_idx >= total_plies)):
                st.session_state.move_idx = stored_idx + 1
                st.rerun()
        with nav_cols[3]:
            if st.button("⏭", help="Go to end",
                         use_container_width=True, disabled=(stored_idx >= total_plies)):
                st.session_state.move_idx = total_plies
                st.rerun()

        # Move indicator
        st.markdown(
            f"<div style='text-align:center; font-size:0.85em; margin:4px 0 8px'>"
            f"<b>{mover}'s move {fullmove}</b>"
            f" · Ply {stored_idx}/{total_plies}"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.divider()

        # ── State 1: awaiting guess ───────────────────────────────────────
        if not st.session_state.submitted:
            st.markdown("**🎯 Guess the average Elo**")
            guess = st.number_input(
                "Your Elo estimate",
                min_value=800,
                max_value=2200,
                value=int(st.session_state.user_guess),
                step=50,
            )
            st.session_state.user_guess = int(guess)

            if st.button("Submit ✅", type="primary", use_container_width=True):
                st.session_state.submitted = True
                st.rerun()

        # ── State 2: showing results ──────────────────────────────────────
        else:
            actual      = game["avg_elo"]
            user_guess  = st.session_state.user_guess
            user_error  = abs(user_guess - actual)
            model_error = (
                abs(predicted_elo - actual)
                if predicted_elo is not None
                else float("inf")
            )

            # Score update — run exactly once per round
            if not st.session_state.counted:
                if user_error < model_error:
                    st.session_state.wins_user  += 1
                elif user_error > model_error:
                    st.session_state.wins_model += 1
                else:
                    st.session_state.ties += 1
                st.session_state.counted = True

            # Metrics stacked vertically (narrow column)
            st.metric("🎯 Real Elo",   f"{actual:.0f}")
            st.metric("🧠 Your guess", f"{user_guess}",
                      f"{user_guess - actual:+.0f} Elo", delta_color="inverse")
            st.metric(
                "🤖 ML guess",
                f"{predicted_elo:.0f}" if predicted_elo is not None else "n/a",
                f"{predicted_elo - actual:+.0f} Elo" if predicted_elo is not None else "",
                delta_color="inverse",
            )

            # Winner banner (compact)
            if user_error < model_error:
                st.success(f"🏆 You won! ({user_error:.0f} vs {model_error:.0f} Elo error)")
            elif user_error > model_error:
                st.error(f"🤖 Model won. ({user_error:.0f} vs {model_error:.0f} Elo error)")
            else:
                st.info(f"🤝 Tie! Both off by {user_error:.0f} Elo")

            if st.button("▶️ Next game", type="primary", use_container_width=True):
                _new_game(n_games)

    # ── Feature breakdown (full width, below both columns) ────────────────
    if st.session_state.submitted:
        feats = get_features(game_idx)
        if feats:
            with st.expander("📊 Why did the model guess that?"):
                _render_feature_breakdown(feats, game["avg_elo"])


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
