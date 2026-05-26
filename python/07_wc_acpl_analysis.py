"""
07_wc_acpl_analysis.py
----------------------
Stockfish ACPL analysis for all World Championship match games.
 
Reads PGN files from data/raw/wc_matches/ (produced by 06_download_wc_matches.py).
Outputs two CSVs:
  data/processed/wc_acpl_per_game.csv   — one row per game, both white_acpl + black_acpl
  data/processed/wc_acpl_per_match.csv  — aggregated per match
 
Resume-safe: already-completed (match_id, game_num) pairs are skipped on restart.
 
ACPL definition (same as script 05):
    loss = eval(best_move) − eval(actual_move_played)
Both scores at the same depth, from the moving player's POV.
"""
 
import csv
import io
import re
import statistics
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
 
import chess
import chess.pgn
import chess.engine
from tqdm import tqdm
 
 
# ── Config ─────────────────────────────────────────────────────────────────────
 
ROOT    = Path(__file__).resolve().parent.parent
PGN_DIR = ROOT / "data" / "raw"  / "wc_matches"
OUT_GAME  = ROOT / "data" / "processed" / "wc_acpl_per_game.csv"
OUT_MATCH = ROOT / "data" / "processed" / "wc_acpl_per_match.csv"
 
ENGINE_PATH = "stockfish"
DEPTH       = 18
CAP_CP      = 1000   # cap per-move loss (avoids mate-score distortions)
SKIP_PLIES  = 8      # ignore first 8 half-moves (opening theory)
MIN_PLIES   = 20     # skip extremely short games (accidents / forfeits)
 
PREFIXES = ["WorldChamp", "PCAChamp", "FideChamp"]
 
# Years that used the FIDE knockout / Braingames format (not classical WC)
KNOCKOUT_YEARS = {2000, 2004}   # WorldChamp only
 
# Set to None to analyse all matches; set to a list of match IDs (stem of the
# PGN filename, without extension) to restrict to those files for quick tests.
PILOT_MATCHES: list[str] | None = None
 
 
# ── Column definitions ─────────────────────────────────────────────────────────
 
GAME_COLUMNS = [
    "match_id", "year", "prefix", "championship_type",
    "game_num", "event", "white", "black",
    "white_elo", "black_elo", "result", "eco", "opening_name",
    "total_plies", "white_acpl", "black_acpl",
]
 
MATCH_COLUMNS = [
    "match_id", "year", "prefix", "championship_type",
    "champion", "challenger",
    "n_games", "n_decisive", "n_short_draws",
    "mean_white_acpl", "mean_black_acpl", "combined_acpl",
]
 
 
# ── Helpers ────────────────────────────────────────────────────────────────────
 
def championship_type(prefix: str, year: int) -> str:
    if prefix == "PCAChamp":
        return "PCA"
    if prefix == "FideChamp":
        return "FIDE_split"
    # WorldChamp
    if year in KNOCKOUT_YEARS:
        return "Classical_KO"
    return "Main"
 
 
def match_id_for(prefix: str, year: int) -> str:
    return f"{prefix}{year}"
 
 
def parse_elo(headers: chess.pgn.Headers, tag: str) -> int | None:
    try:
        v = int(headers.get(tag, ""))
        return v if v > 0 else None
    except (ValueError, TypeError):
        return None
 
 
def safe_mean(lst: list[float]) -> float | None:
    return round(statistics.mean(lst), 2) if lst else None
 
 
def game_to_pgn_string(game: chess.pgn.Game) -> str:
    buf = io.StringIO()
    print(game, file=buf)
    return buf.getvalue()
 
 
# ── Resume logic ───────────────────────────────────────────────────────────────
 
def load_completed(path: Path) -> set[tuple[str, int]]:
    """Return set of (match_id, game_num) already in the output CSV."""
    if not path.exists():
        return set()
    completed = set()
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                completed.add((row["match_id"], int(row["game_num"])))
            except (KeyError, ValueError):
                pass
    return completed
 
 
# ── Build task list ────────────────────────────────────────────────────────────
 
def build_tasks(completed: set[tuple[str, int]]) -> list[tuple]:
    """
    Enumerate all games across all PGN files.
    Each task is a tuple:
        (pgn_str, match_id, year, prefix, champ_type, game_num, headers_dict)
    """
    tasks = []
 
    for pgn_path in sorted(PGN_DIR.glob("*.pgn")):
        stem = pgn_path.stem   # e.g. "WorldChamp1972"
 
        if PILOT_MATCHES is not None and stem not in PILOT_MATCHES:
            continue
 
        # Identify prefix
        prefix = None
        for p in sorted(PREFIXES, key=len, reverse=True):
            if stem.startswith(p):
                prefix = p
                break
        if prefix is None:
            continue
 
        try:
            year = int(stem[len(prefix):])
        except ValueError:
            continue
 
        mid      = match_id_for(prefix, year)
        champ_ty = championship_type(prefix, year)
 
        with open(pgn_path, encoding="utf-8", errors="replace") as f:
            raw = f.read()
 
        # Some PGN files omit the required blank line between games.
        # chess.pgn needs it to detect game boundaries correctly.
        raw = re.sub(r'(1/2-1/2|1-0|0-1|\*)([ \t]*)\r?\n\[',
                     r'\1\2\n\n[', raw)
        pgn_io = io.StringIO(raw)
 
        game_num = 0
        while True:
            game = chess.pgn.read_game(pgn_io)
            if game is None:
                break
 
            game_num += 1
 
            if (mid, game_num) in completed:
                continue   # already processed — skip
 
            h = game.headers
            headers_dict = {
                "event":        h.get("Event", ""),
                "white":        h.get("White", ""),
                "black":        h.get("Black", ""),
                "white_elo":    parse_elo(h, "WhiteElo"),
                "black_elo":    parse_elo(h, "BlackElo"),
                "result":       h.get("Result", ""),
                "eco":          h.get("ECO", ""),
                "opening_name": h.get("Opening", ""),
            }
 
            pgn_str = game_to_pgn_string(game)
            tasks.append((pgn_str, mid, year, prefix, champ_ty, game_num, headers_dict))
 
    return tasks
 
 
# ── ACPL worker ────────────────────────────────────────────────────────────────
 
def acpl_both_colors(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    moves: list[chess.Move],
) -> tuple[list[float], list[float]]:
    """
    Single-pass ACPL for both colors simultaneously.
    Returns (white_losses, black_losses) — lists of per-move centipawn losses.
    """
    white_losses: list[float] = []
    black_losses: list[float] = []
 
    for ply_idx, actual_move in enumerate(moves):
        current_color = board.turn
 
        if ply_idx >= SKIP_PLIES:
            info = engine.analyse(board, chess.engine.Limit(depth=DEPTH))
            pv   = info.get("pv", [])
 
            if pv:
                best_move  = pv[0]
                best_score = info["score"].pov(current_color).score(mate_score=10_000)
 
                if actual_move == best_move:
                    loss = 0
                else:
                    board.push(actual_move)
                    info_after   = engine.analyse(board, chess.engine.Limit(depth=DEPTH))
                    actual_score = info_after["score"].pov(current_color).score(mate_score=10_000)
                    board.pop()
                    loss = max(0, min(CAP_CP, best_score - actual_score))
 
                if current_color == chess.WHITE:
                    white_losses.append(loss)
                else:
                    black_losses.append(loss)
 
        board.push(actual_move)
 
    return white_losses, black_losses
 
 
def analyze_game_worker(args: tuple) -> dict | None:
    """
    Subprocess worker.  Spawns its own Stockfish instance per call.
    Returns a dict matching GAME_COLUMNS, or None on failure.
    """
    pgn_str, match_id, year, prefix, champ_ty, game_num, hd = args
 
    game  = chess.pgn.read_game(io.StringIO(pgn_str))
    board = game.board()
    moves = list(game.mainline_moves())
 
    if len(moves) < MIN_PLIES:
        # Game too short to be meaningful (forfeit, accident, etc.)
        return {
            "match_id":          match_id,
            "year":              year,
            "prefix":            prefix,
            "championship_type": champ_ty,
            "game_num":          game_num,
            **hd,
            "total_plies":  len(moves),
            "white_acpl":   None,
            "black_acpl":   None,
        }
 
    try:
        engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
        white_losses, black_losses = acpl_both_colors(engine, board, moves)
        engine.quit()
 
        return {
            "match_id":          match_id,
            "year":              year,
            "prefix":            prefix,
            "championship_type": champ_ty,
            "game_num":          game_num,
            **hd,
            "total_plies": len(moves),
            "white_acpl":  safe_mean(white_losses),
            "black_acpl":  safe_mean(black_losses),
        }
 
    except Exception as exc:
        try:
            engine.quit()
        except Exception:
            pass
        print(f"\n  [ERROR] {match_id} game {game_num}: {exc}")
        return None
 
 
# ── Run analysis ───────────────────────────────────────────────────────────────
 
def run_analysis(
    tasks: list[tuple],
    n_workers: int,
    csv_writer: csv.DictWriter,
    csv_file,
) -> list[dict]:
    results = []
    with Pool(processes=n_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(analyze_game_worker, tasks),
            total=len(tasks),
            desc="WC games",
            unit="game",
        ):
            if result is not None:
                results.append(result)
                csv_writer.writerow(result)
                csv_file.flush()
    return results
 
 
# ── Per-match aggregation ──────────────────────────────────────────────────────
 
def is_short_draw(row: dict) -> bool:
    """Draw in a game with fewer than 40 plies (20 moves per side)."""
    return row["result"] == "1/2-1/2" and (row["total_plies"] or 0) < 40
 
 
def aggregate_matches(game_rows: list[dict]) -> list[dict]:
    """Group game rows by match_id and compute per-match summary."""
    by_match: dict[str, list[dict]] = defaultdict(list)
    for row in game_rows:
        by_match[row["match_id"]].append(row)
 
    match_rows = []
    for mid, rows in sorted(by_match.items()):
        r0 = rows[0]   # all rows share the same match-level metadata
 
        white_acpls = [r["white_acpl"] for r in rows if r["white_acpl"] is not None]
        black_acpls = [r["black_acpl"] for r in rows if r["black_acpl"] is not None]
        all_acpls   = white_acpls + black_acpls
 
        n_decisive    = sum(1 for r in rows if r["result"] in ("1-0", "0-1"))
        n_short_draws = sum(1 for r in rows if is_short_draw(r))
 
        match_rows.append({
            "match_id":          mid,
            "year":              r0["year"],
            "prefix":            r0["prefix"],
            "championship_type": r0["championship_type"],
            "champion":          "",      # PGN Mentor doesn't tag champion; left blank
            "challenger":        "",
            "n_games":           len(rows),
            "n_decisive":        n_decisive,
            "n_short_draws":     n_short_draws,
            "mean_white_acpl":   safe_mean(white_acpls),
            "mean_black_acpl":   safe_mean(black_acpls),
            "combined_acpl":     safe_mean(all_acpls),
        })
 
    return match_rows
 
 
# ── Summary printout ───────────────────────────────────────────────────────────
 
def _print_acpl_block(game_rows: list[dict], indent: str = "  ") -> None:
    """Print Mean / Median / Std-dev ACPL table for a set of game rows."""
    w = [r["white_acpl"] for r in game_rows if r["white_acpl"] is not None]
    b = [r["black_acpl"] for r in game_rows if r["black_acpl"] is not None]
    if not w or not b:
        print(f"{indent}  (no data)")
        return
    print(f"{indent}{'Metric':<28} {'White':>10} {'Black':>10}")
    print(f"{indent}{'─' * 50}")
    for label, fn in [
        ("Mean ACPL",    statistics.mean),
        ("Median ACPL",  statistics.median),
        ("Std dev ACPL", statistics.stdev),
    ]:
        try:
            print(f"{indent}{label:<28} {fn(w):>10.1f} {fn(b):>10.1f}")
        except statistics.StatisticsError:
            print(f"{indent}{label:<28} {'—':>10} {'—':>10}")
 
 
def print_summary(game_rows: list[dict], match_rows: list[dict]) -> None:
    W = 62
    print(f"\n{'═' * W}")
    print(f"  Games analysed : {len(game_rows)}")
    print(f"  Matches covered: {len(match_rows)}")
 
    # ── Overall stats ─────────────────────────────────────────────
    print(f"{'─' * W}")
    print(f"  OVERALL")
    print(f"{'─' * W}")
    _print_acpl_block(game_rows)
 
    # ── Per-match stats ───────────────────────────────────────────
    by_match: dict[str, list[dict]] = defaultdict(list)
    for r in game_rows:
        by_match[r["match_id"]].append(r)
 
    for mid, rows in sorted(by_match.items(), key=lambda kv: kv[1][0]["year"]):
        year = rows[0]["year"]
        n    = len(rows)
        print(f"{'─' * W}")
        print(f"  {mid}  ({year})  —  {n} game(s)")
        print(f"{'─' * W}")
        _print_acpl_block(rows)
 
    # ── Year range ────────────────────────────────────────────────
    if match_rows:
        earliest = min(match_rows, key=lambda r: r["year"])
        latest   = max(match_rows, key=lambda r: r["year"])
        print(f"{'─' * W}")
        print(f"  Year range: {earliest['year']} ({earliest['match_id']}) "
              f"\u2192 {latest['year']} ({latest['match_id']})")
 
    print(f"{'═' * W}")
 
# ── Main ───────────────────────────────────────────────────────────────────────
 
def main() -> None:
    n_workers = max(1, cpu_count() - 1)
    print(f"Workers: {n_workers}  |  Depth: {DEPTH}  |  "
          f"Cap: {CAP_CP} cp  |  Skip: {SKIP_PLIES} plies")
 
    # ── Resume: what's already done? ──────────────────────────────────────────
    completed = load_completed(OUT_GAME)
    print(f"Already completed: {len(completed)} game(s)")
 
    # ── Build task list ───────────────────────────────────────────────────────
    tasks = build_tasks(completed)
    print(f"Games to analyse : {len(tasks)}")
 
    if not tasks:
        print("Nothing new to analyse.")
    else:
        est_min = len(tasks) * 3
        est_max = len(tasks) * 5
        print(f"Estimated runtime: {est_min}–{est_max} min  "
              f"(~3–5 min/game at depth {DEPTH})\n")
 
        OUT_GAME.parent.mkdir(parents=True, exist_ok=True)
 
        # Open in append mode so resume works correctly
        file_exists = OUT_GAME.exists()
        with open(OUT_GAME, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=GAME_COLUMNS)
            if not file_exists:
                writer.writeheader()
                csv_file.flush()
 
            new_results = run_analysis(tasks, n_workers, writer, csv_file)
 
        print(f"\n{len(new_results)} new game(s) written → {OUT_GAME}")
 
    # ── Load full game CSV for aggregation ────────────────────────────────────
    all_game_rows: list[dict] = []
    if OUT_GAME.exists():
        with open(OUT_GAME, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Coerce numeric fields
                for col in ("year", "game_num", "total_plies"):
                    try:
                        row[col] = int(row[col])
                    except (ValueError, TypeError):
                        row[col] = None
                for col in ("white_acpl", "black_acpl"):
                    try:
                        row[col] = float(row[col]) if row[col] not in ("", "None") else None
                    except (ValueError, TypeError):
                        row[col] = None
                all_game_rows.append(row)
 
    # ── Aggregate per match and save ──────────────────────────────────────────
    match_rows = aggregate_matches(all_game_rows)
    with open(OUT_MATCH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MATCH_COLUMNS)
        writer.writeheader()
        writer.writerows(match_rows)
    print(f"{len(match_rows)} match rows written → {OUT_MATCH}")
 
    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary(all_game_rows, match_rows)
 
 
if __name__ == "__main__":
    main()
 