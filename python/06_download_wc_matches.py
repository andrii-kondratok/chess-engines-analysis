"""
06_download_wc_matches.py
-------------------------
Download all available World Championship PGN files from PGN Mentor.

URL pattern: https://www.pgnmentor.com/events/{Prefix}{Year}.pgn

Three prefixes:
  WorldChamp — main line (1886–1992, reunification 2006+)
  PCAChamp   — Kasparov's PCA breakaway (1993, 1995)
  FideChamp  — FIDE championship during 1993–2006 split

Strategy: try every prefix × year 1886–2024, keep 200 responses
          that look like actual PGN (contain "[Event"). 404s silently skipped.
"""

import time
import chess.pgn
import io
import requests
from pathlib import Path
from collections import defaultdict


# ── Config ─────────────────────────────────────────────────────────────────────

ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "raw" / "wc_matches"

BASE_URL   = "https://www.pgnmentor.com/events/{prefix}{year}.pgn"
PREFIXES   = ["WorldChamp", "PCAChamp", "FideChamp"]
YEAR_RANGE = range(1886, 2025)

SLEEP_SEC  = 0.3
TIMEOUT    = 10
MIN_BYTES  = 100


# ── Download ───────────────────────────────────────────────────────────────────

def try_download(session: requests.Session, prefix: str, year: int) -> bool:
    """Attempt to fetch one PGN file. Returns True if saved, False otherwise."""
    url = BASE_URL.format(prefix=prefix, year=year)
    try:
        r = session.get(url, timeout=TIMEOUT)
    except requests.RequestException:
        return False

    if r.status_code != 200:
        return False
    if len(r.content) < MIN_BYTES:
        return False
    if b"[Event" not in r.content:   # reject HTML error pages
        return False

    out_path = OUT_DIR / f"{prefix}{year}.pgn"
    out_path.write_bytes(r.content)
    return True


def download_all() -> list[dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = "chess-research-bot/1.0 (academic)"

    downloaded = []
    total_requests = len(PREFIXES) * len(YEAR_RANGE)
    done = 0

    for prefix in PREFIXES:
        for year in YEAR_RANGE:
            done += 1
            success = try_download(session, prefix, year)
            if success:
                print(f"  ✓  {prefix}{year}.pgn")
                downloaded.append({"prefix": prefix, "year": year,
                                   "filename": f"{prefix}{year}.pgn"})
            time.sleep(SLEEP_SEC)

    return downloaded


# ── Download summary ───────────────────────────────────────────────────────────

def print_download_summary(downloaded: list[dict]) -> None:
    print(f"\n{'═' * 50}")
    print(f"  Downloaded: {len(downloaded)} files")
    print(f"{'─' * 50}")

    by_prefix = defaultdict(list)
    for d in downloaded:
        by_prefix[d["prefix"]].append(d["year"])

    for prefix in PREFIXES:
        years = sorted(by_prefix[prefix])
        if not years:
            print(f"  {prefix:<14}  0 files")
            continue

        # Find gaps
        gaps = [y for y, yn in zip(years, years[1:]) if yn - y > 1]
        gap_str = f"  gaps: {gaps}" if gaps else "  no gaps"
        print(f"  {prefix:<14}  {len(years):2d} files  "
              f"({years[0]}–{years[-1]}){gap_str}")

    all_years = sorted(d["year"] for d in downloaded)
    if all_years:
        print(f"{'─' * 50}")
        print(f"  Overall coverage: {all_years[0]}–{all_years[-1]}")
    print(f"{'═' * 50}")


# ── Sanity check ───────────────────────────────────────────────────────────────

def scan_pgn_file(path: Path) -> dict:
    """Count games and extract first game's White/Black from a PGN file."""
    games = []
    with open(path, encoding="utf-8", errors="replace") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            games.append(game)

    first_white = first_black = "—"
    if games:
        first_white = games[0].headers.get("White", "—")
        first_black = games[0].headers.get("Black", "—")

    return {
        "filename":    path.name,
        "games_count": len(games),
        "first_white": first_white,
        "first_black": first_black,
    }


def run_sanity_check() -> None:
    pgn_files = sorted(OUT_DIR.glob("*.pgn"))
    if not pgn_files:
        print("\nNo PGN files found in output directory.")
        return

    print(f"\nScanning {len(pgn_files)} files...\n")

    rows = []
    for path in pgn_files:
        # Parse prefix and year from filename
        name = path.stem   # e.g. "WorldChamp1886"
        for prefix in sorted(PREFIXES, key=len, reverse=True):
            if name.startswith(prefix):
                year = name[len(prefix):]
                break
        else:
            prefix, year = "?", "?"

        info = scan_pgn_file(path)
        rows.append({
            "filename":    info["filename"],
            "year":        year,
            "prefix":      prefix,
            "games":       info["games_count"],
            "first_white": info["first_white"][:22],
            "first_black": info["first_black"][:22],
        })

    # Print as aligned table
    col_w = [34, 6, 12, 6, 24, 24]
    headers = ["filename", "year", "prefix", "games", "first_white", "first_black"]
    header_row = "  ".join(h.ljust(w) for h, w in zip(headers, col_w))
    print(header_row)
    print("─" * len(header_row))
    for row in rows:
        values = [str(row[h]) for h in headers]
        print("  ".join(v.ljust(w) for v, w in zip(values, col_w)))

    total_games = sum(r["games"] for r in rows)
    print(f"\nTotal games across all files: {total_games}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Trying {len(PREFIXES)} prefixes × {len(YEAR_RANGE)} years "
          f"= {len(PREFIXES) * len(YEAR_RANGE)} requests")
    print(f"Estimated time: ~{len(PREFIXES) * len(YEAR_RANGE) * SLEEP_SEC / 60:.1f} min\n")

    downloaded = download_all()
    print_download_summary(downloaded)
    run_sanity_check()


if __name__ == "__main__":
    main()
