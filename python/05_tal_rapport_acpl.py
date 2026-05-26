"""
05_tal_rapport_acpl.py
----------------------
Tal (1980, Elo 2705) vs Rapport (2017, Elo 2707) — ACPL comparison via Stockfish 18.

Tests Regan & Haworth (2011): equal FIDE Elo implies equal objective skill across eras.
If true, ACPL should be statistically indistinguishable between the two players.
"""

import io
import csv
import statistics
from pathlib import Path
from multiprocessing import Pool, cpu_count

import chess
import chess.pgn
import chess.engine
from tqdm import tqdm


# ── Config ─────────────────────────────────────────────────────────────────────

ROOT      = Path(__file__).resolve().parent.parent
RAW_DIR   = ROOT / "data" / "raw"
OUT_FILE  = ROOT / "data" / "processed" / "tal_rapport_acpl.csv"

ENGINE_PATH     = "stockfish"
DEPTH           = 18
CAP_CP          = 1000   # cap per-move loss to avoid mate score distortions
SKIP_PLIES      = 8      # ignore first 8 half-moves (opening theory)
MIN_OPPONENT_ELO = 2500

TAL_FILES     = ["tal_games_1.pgn", "tal_games_2.pgn", "tal_games_3.pgn"]
RAPPORT_FILES = ["rapport_games_1.pgn", "rapport_games_2.pgn",
                 "rapport_games_3.pgn", "rapport_games_4.pgn"]

MIN_TC_SECS   = 600   # Rapport only: skip games faster than this base time


# ── Step 1 — Load PGN files ────────────────────────────────────────────────────

def load_pgn_files(filenames: list[str]) -> list[chess.pgn.Game]:
    games = []
    for fname in filenames:
        path = RAW_DIR / fname
        with open(path, encoding="utf-8", errors="replace") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                games.append(game)
    return games


# ── Step 2 — Filter games ──────────────────────────────────────────────────────

def parse_year(headers: chess.pgn.Headers) -> int | None:
    date = headers.get("Date", "")
    try:
        return int(date.split(".")[0])
    except (ValueError, AttributeError):
        return None


def parse_base_time(tc: str) -> int | None:
    """Return base time in seconds from a TimeControl string. None if unparseable."""
    if not tc or tc in ("-", "?"):
        return None
    try:
        # "600+5"  →  600
        # "600"    →  600
        # "40/5400+30"  →  last segment base
        segment = tc.split("/")[-1]   # handles "40/5400" style
        return int(segment.split("+")[0])
    except (ValueError, IndexError):
        return None


def find_player_color(headers: chess.pgn.Headers, name_fragment: str) -> chess.Color | None:
    """Find which color the target player played, by partial name match."""
    white = headers.get("White", "").lower()
    black = headers.get("Black", "").lower()
    frag  = name_fragment.lower()
    if frag in white:
        return chess.WHITE
    if frag in black:
        return chess.BLACK
    return None


def get_opponent_elo(headers: chess.pgn.Headers, player_color: chess.Color) -> int | None:
    tag = "BlackElo" if player_color == chess.WHITE else "WhiteElo"
    try:
        elo = int(headers.get(tag, ""))
        return elo if elo > 0 else None
    except (ValueError, TypeError):
        return None


def filter_games(
    games: list[chess.pgn.Game],
    player_fragment: str,
    check_time_control: bool = False,
) -> tuple[list[tuple], list[tuple]]:
    """
    Returns (kept, rejected) where each element is (game, metadata_dict).
    metadata_dict for kept games includes color, opponent_elo, and year.
    Year is extracted for the CSV output only — not used for filtering.
    """
    kept     = []
    rejected = []

    for game in games:
        h    = game.headers
        year = parse_year(h)   # None if tag missing — kept as-is in output

        color = find_player_color(h, player_fragment)
        if color is None:
            rejected.append((game, f"{player_fragment} not found in headers"))
            continue

        opp_elo = get_opponent_elo(h, color)
        if opp_elo is None:
            rejected.append((game, "opponent Elo missing or invalid"))
            continue

        if opp_elo < MIN_OPPONENT_ELO:
            rejected.append((game, f"opponent Elo {opp_elo} < {MIN_OPPONENT_ELO}"))
            continue

        if check_time_control:
            tc   = h.get("TimeControl", "")
            base = parse_base_time(tc)
            if base is not None and base < MIN_TC_SECS:
                rejected.append((game, f"too fast: {tc} (base={base}s)"))
                continue

        kept.append((game, {"color": color, "opponent_elo": opp_elo, "year": year}))

    return kept, rejected


def print_filter_report(player: str, kept: list, rejected: list) -> None:
    print(f"\n{player}: {len(kept)} kept, {len(rejected)} rejected")
    # Tally rejection reasons
    reasons: dict[str, int] = {}
    for _, reason in rejected:
        reasons[reason if not reason[0].isdigit() else reason.split()[0] + "…"] \
            = reasons.get(reason, 0) + 1
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  [{count:3d}]  {reason}")


# ── Step 3 — Stockfish ACPL analysis ──────────────────────────────────────────

def analyze_game_worker(args: tuple) -> dict | None:
    """
    Worker function — runs in a subprocess, spawns its own Stockfish instance.
    Receives a game as a PGN string to survive pickling.

    True ACPL definition:
        loss = eval(best_move) - eval(actual_move_played)
    Both scores are from the target player's POV at the same depth.
    If actual == best, loss = 0 and the second engine call is skipped.
    """
    pgn_str, player_name, player_color, meta = args

    game  = chess.pgn.read_game(io.StringIO(pgn_str))
    h     = game.headers
    board = game.board()
    moves = list(game.mainline_moves())

    try:
        engine    = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
        cp_losses = []

        for ply_idx, actual_move in enumerate(moves):
            is_target_move = (board.turn == player_color) and (ply_idx >= SKIP_PLIES)

            if is_target_move:
                # ── Evaluate current position to find best move + its score ──
                info = engine.analyse(board, chess.engine.Limit(depth=DEPTH))
                pv   = info.get("pv", [])

                if pv:  # pv is empty only in terminal positions — skip those
                    best_move  = pv[0]
                    best_score = info["score"].pov(player_color).score(mate_score=10_000)

                    if actual_move == best_move:
                        # Perfect move — no loss, no second engine call needed
                        loss = 0
                    else:
                        # Push actual move temporarily, evaluate the result
                        board.push(actual_move)
                        info_after   = engine.analyse(board, chess.engine.Limit(depth=DEPTH))
                        # After player's move it's the opponent's turn, but
                        # .pov(player_color) keeps the perspective consistent
                        actual_score = info_after["score"].pov(player_color).score(mate_score=10_000)
                        board.pop()  # restore — main push happens below

                        loss = max(0, min(CAP_CP, best_score - actual_score))

                    cp_losses.append(loss)

            # Advance the board for all plies (whether we analysed or not)
            board.push(actual_move)

        engine.quit()

        acpl      = round(statistics.mean(cp_losses), 2) if cp_losses else None
        color_str = "White" if player_color == chess.WHITE else "Black"

        return {
            "player":       player_name,
            "year":         meta["year"],
            "event":        h.get("Event", ""),
            "white":        h.get("White", ""),
            "black":        h.get("Black", ""),
            "opponent_elo": meta["opponent_elo"],
            "result":       h.get("Result", ""),
            "total_plies":  len(moves),
            "acpl":         acpl,
            "color":        color_str,
        }

    except Exception as exc:
        try:
            engine.quit()
        except Exception:
            pass
        print(f"\n  [ERROR] {player_name} game failed: {exc}")
        return None


def game_to_pgn_string(game: chess.pgn.Game) -> str:
    buf = io.StringIO()
    print(game, file=buf)
    return buf.getvalue()


COLUMNS = ["player", "year", "event", "white", "black",
           "opponent_elo", "result", "total_plies", "acpl", "color"]


def run_analysis(
    kept: list[tuple],
    player_name: str,
    n_workers: int,
    csv_writer: csv.DictWriter,
    csv_file,
) -> list[dict]:
    """
    Run ACPL analysis over all kept games using a process pool.
    Each result is written to CSV immediately after the game finishes —
    partial results survive if the script crashes mid-run.
    """
    tasks = [
        (game_to_pgn_string(game), player_name, meta["color"], meta)
        for game, meta in kept
    ]

    results = []
    with Pool(processes=n_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(analyze_game_worker, tasks),
            total=len(tasks),
            desc=player_name,
            unit="game",
        ):
            if result is not None:
                results.append(result)
                csv_writer.writerow(result)
                csv_file.flush()   # hit disk now, don't wait for buffer

    return results


# ── Step 5 — Summary statistics ───────────────────────────────────────────────

def print_summary(all_results: list[dict]) -> None:
    # Build per-player stats once, store in a plain dict
    stats: dict[str, dict] = {}
    for player in ("Tal", "Rapport"):
        rows  = [r for r in all_results if r["player"] == player]
        stats[player] = {
            "rows":  rows,
            "acpls": [r["acpl"] for r in rows if r["acpl"] is not None],
            "elos":  [r["opponent_elo"] for r in rows],
        }

    metrics: list[tuple[str, str, any]] = [
        ("Games analyzed",  "n",    None),
        ("Mean opp. Elo",   "elo",  statistics.mean),
        ("Median opp. Elo", "elo",  statistics.median),
        ("Mean ACPL",       "acpl", statistics.mean),
        ("Median ACPL",     "acpl", statistics.median),
        ("Std dev ACPL",    "acpl", statistics.stdev),
    ]

    print("\n" + "═" * 58)
    print(f"{'Metric':<28} {'Tal':>12} {'Rapport':>12}")
    print("─" * 58)

    for label, field, fn in metrics:
        row = f"{label:<28}"
        for player in ("Tal", "Rapport"):
            s    = stats[player]
            data = s["acpls"] if field == "acpl" else s["elos"]
            if field == "n":
                row += f"{len(s['rows']):>12}"
            elif data and fn:
                try:
                    row += f"{fn(data):>12.1f}"
                except statistics.StatisticsError:
                    row += f"{'—':>12}"
            else:
                row += f"{'—':>12}"
        print(row)

    print("═" * 58)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:

    n_workers = max(1, cpu_count() - 1)
    print(f"Workers: {n_workers}  |  Depth: {DEPTH}  |  Cap: {CAP_CP} cp  |  Skip: {SKIP_PLIES} plies")

    # ── Step 1: load ──────────────────────────────────────────────────────────
    print("\n── Loading PGN files ───────────────────────────────────────────")
    tal_games     = load_pgn_files(TAL_FILES)
    rapport_games = load_pgn_files(RAPPORT_FILES)
    print(f"  Tal     : {len(tal_games)} games loaded")
    print(f"  Rapport : {len(rapport_games)} games loaded")

    # ── Step 2: filter ────────────────────────────────────────────────────────
    print("\n── Filtering games ─────────────────────────────────────────────")
    tal_kept, tal_rejected         = filter_games(tal_games,     "Tal",     check_time_control=False)
    rapport_kept, rapport_rejected = filter_games(rapport_games, "Rapport", check_time_control=True)
    print_filter_report("Tal",     tal_kept,     tal_rejected)
    print_filter_report("Rapport", rapport_kept, rapport_rejected)

    total_games = len(tal_kept) + len(rapport_kept)
    print(f"\nTotal games to analyse: {total_games}")
    print(f"Estimated runtime at depth {DEPTH}: {total_games * 3:.0f}–{total_games * 5:.0f} min"
          f"  (~3–5 min/game; 2× calls per target-player move)")

    # ── Step 3 & 4: analyse + write each result immediately ───────────────────
    print("\n── Stockfish analysis ──────────────────────────────────────────")
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=COLUMNS)
        writer.writeheader()
        csv_file.flush()

        tal_results     = run_analysis(tal_kept,     "Tal",     n_workers, writer, csv_file)
        rapport_results = run_analysis(rapport_kept, "Rapport", n_workers, writer, csv_file)

    all_results = tal_results + rapport_results
    print(f"\n{len(all_results)} games written → {OUT_FILE}")

    # ── Step 5: summary ───────────────────────────────────────────────────────
    print_summary(all_results)


if __name__ == "__main__":
    main()
