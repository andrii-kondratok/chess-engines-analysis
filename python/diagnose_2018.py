"""
diagnose_2018.py
----------------
Parse WorldChamp2018.pgn move by move using python-chess.
For every game that fails, print:
  - Round / players / date
  - Move number and the failing move (as it appears in the PGN)
  - Full FEN of the position immediately before that move
  - Last 5 valid moves played before the failure
"""

from pathlib import Path
import chess
import chess.pgn

PGN_PATH = (
    Path(__file__).resolve().parent.parent
    / "data" / "raw" / "wc_matches" / "WorldChamp2018.pgn"
)


def move_label(ply_idx: int, move_san: str) -> str:
    """Return a human-readable label like '3. e4' or '3... e5'."""
    move_num = ply_idx // 2 + 1
    dots     = "." if ply_idx % 2 == 0 else "..."
    return f"{move_num}{dots} {move_san}"


def diagnose_game(game_idx: int, game: chess.pgn.Game) -> bool:
    """
    Walk every move in the game, catching illegal-move errors.
    Returns True if the game parsed cleanly, False if it failed.
    """
    h = game.headers
    tag = (
        f"Game {game_idx}  |  "
        f"Round {h.get('Round', '?')}  |  "
        f"{h.get('White', '?')} vs {h.get('Black', '?')}  |  "
        f"Date {h.get('Date', '?')}"
    )

    board      = game.board()
    history    : list[str] = []   # SAN of moves played so far
    node       = game

    while node.variations:
        next_node   = node.variations[0]
        pgn_move    = next_node.move          # chess.Move object
        ply_idx     = board.ply()

        # Attempt to convert to SAN *before* pushing (SAN needs the pre-move board)
        try:
            san = board.san(pgn_move)
        except (chess.IllegalMoveError, chess.AmbiguousMoveError, AssertionError):
            san = str(pgn_move)   # fallback to UCI if SAN itself blows up

        # Check legality
        if pgn_move not in board.legal_moves:
            print("=" * 70)
            print(f"FAIL  {tag}")
            print(f"  Failing move : {move_label(ply_idx, san)}")
            print(f"  FEN before   : {board.fen()}")
            tail = history[-5:]
            print(f"  Last ≤5 moves: {' | '.join(tail) if tail else '(none)'}")
            print("=" * 70)
            return False

        history.append(move_label(ply_idx, san))
        board.push(pgn_move)
        node = next_node

    return True


def main() -> None:
    with open(PGN_PATH, encoding="utf-8", errors="replace") as f:
        game_idx  = 0
        ok_count  = 0
        bad_count = 0

        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            game_idx += 1

            if diagnose_game(game_idx, game):
                ok_count += 1
            else:
                bad_count += 1

    print()
    print(f"Total games : {game_idx}")
    print(f"  OK        : {ok_count}")
    print(f"  Failed    : {bad_count}")


if __name__ == "__main__":
    main()
