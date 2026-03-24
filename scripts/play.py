"""Play against a trained chess model.

Usage:
    python scripts/play.py --ckpt path/to/checkpoint.ckpt [--eval] [--top-n 5]
"""

import argparse
import contextlib
import operator
import sys

import chess
import rootutils
import torch as T
import torch.nn.functional as F

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

from src.datamodules import encode_board
from src.networks import ChessModel

DIVIDER = "─" * 44


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play against the chess model.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to a Lightning checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run the model on (cpu, cuda, auto).",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Whether to show model evaluation of each position.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=0,
        help="How many top moves to show in the evaluation (0 for all).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def choose_device(preference: str) -> T.device:
    if preference == "auto":
        return T.device("cuda" if T.cuda.is_available() else "cpu")
    return T.device(preference)


def load_model(ckpt_path: str, device: T.device) -> ChessModel:
    model = ChessModel.load_from_checkpoint(
        ckpt_path, map_location=device, weights_only=False
    )
    model.eval()
    model.to(device)
    return model


@T.no_grad()
def evaluate_moves(model: ChessModel, board: chess.Board, device: T.device):
    """Score every legal move and return a list sorted best → worst for white.

    Each entry is (move, win%, draw%, loss%) where "win" means white wins.
    """
    scored: list[tuple[chess.Move, float, float, float]] = []

    for move in board.legal_moves:
        board.push(move)

        if board.is_checkmate():
            # White just moved and it's checkmate → white wins with certainty.
            probs = (1.0, 0.0, 0.0)
        elif board.is_stalemate() or board.is_insufficient_material():
            probs = (0.0, 1.0, 0.0)
        else:
            enc_board = encode_board(board)
            enc_board = T.from_numpy(enc_board).unsqueeze(0).to(device)
            logits = model(enc_board)  # (1, 3)
            p = F.softmax(logits, dim=-1).squeeze(0)  # (3,)
            probs = (p[0].item(), p[1].item(), p[2].item())

        scored.append((move, *probs))
        board.pop()

    # Sort by white win probability (descending).
    scored.sort(key=operator.itemgetter(1), reverse=True)
    return scored


def show_board(board: chess.Board) -> None:
    print()
    print("     a  b  c  d  e  f  g  h")
    print("   ┌────────────────────────┐")
    for rank in range(7, -1, -1):
        row = f" {rank + 1} │"
        for file in range(8):
            piece = board.piece_at(chess.square(file, rank))
            symbol = piece.unicode_symbol() if piece else "·"
            row += f" {symbol} "
        row += f"│ {rank + 1}"
        print(row)
    print("   └────────────────────────┘")
    print("     a  b  c  d  e  f  g  h")
    print()


def show_eval_table(scored, top_n: int) -> None:
    n = len(scored) if top_n == 0 else min(top_n, len(scored))
    print(DIVIDER)
    print(f"  {'Move':<8} {'Win':>7} {'Draw':>7} {'Loss':>7}")
    print(DIVIDER)
    for move, win, draw, loss in scored[:n]:
        bar_len = int(win * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(
            f" {move!s:<8}"
            f" {win * 100:>6.1f}%"
            f" {draw * 100:>6.1f}%"
            f" {loss * 100:>6.1f}%"
            f" {bar}"
        )
    if n < len(scored):
        print(f"  … and {len(scored) - n} more moves")
    print(DIVIDER)


def print_help() -> None:
    print("""
Commands:
  <move>     Enter a move in UCI (e.g. e7e5) or SAN (e.g. e5, Nf6).
  undo       Undo the last full move (your move + model's move).
  board      Redisplay the board.
  eval       Toggle evaluation display on/off.
  moves      Show all your legal moves.
  quit       Exit the game.
  help       Show this message.
""")


def play(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    print(f"Using device: {device}")
    print(f"Loading checkpoint: {args.ckpt}")

    model = load_model(args.ckpt, device)
    board = chess.Board()
    show_eval = args.eval
    top_n = args.top_n

    # History of FENs for undo support.
    history: list[str] = [board.fen()]

    print("\n♟  Chess Bot — Interactive Play  ♟")
    print("Model plays as WHITE.  You play as BLACK.")
    print('Type "help" for a list of commands.\n')

    show_board(board)

    while not board.is_game_over():
        # ── Model's turn (White) ─────────────────────────────────
        if board.turn == chess.WHITE:
            print("Model is thinking …")
            scored = evaluate_moves(model, board, device)

            if show_eval:
                show_eval_table(scored, top_n)

            best_move, win, draw, loss = scored[0]
            board.push(best_move)
            history.append(board.fen())

            print(
                f"Model plays: {best_move}  "
                f"(W {win * 100:.1f}% / D {draw * 100:.1f}% / L {loss * 100:.1f}%)"
            )
            show_board(board)

            if board.is_game_over():
                break
            continue

        # ── Human's turn (Black) ─────────────────────────────────
        raw = input("Your move ▸ ").strip()
        if not raw:
            continue

        cmd = raw.lower()

        if cmd in {"quit", "exit", "q"}:
            print("Thanks for playing!")
            sys.exit(0)
        if cmd == "help":
            print_help()
            continue
        if cmd == "board":
            show_board(board)
            continue
        if cmd == "eval":
            show_eval = not show_eval
            print(f"Evaluation display {'ON' if show_eval else 'OFF'}.")
            continue
        if cmd == "moves":
            legal = sorted(str(m) for m in board.legal_moves)
            print("Legal moves:", ", ".join(legal))
            continue
        if cmd == "undo":
            if len(history) < 3:
                print("Nothing to undo.")
                continue
            history.pop()  # undo model's reply
            history.pop()  # undo player's move
            board.set_fen(history[-1])
            print("Undone last full move.")
            show_board(board)
            continue

        # Try parsing as UCI, then SAN.
        move: chess.Move | None = None
        with contextlib.suppress(ValueError):
            move = board.parse_uci(raw)
        if move is None:
            with contextlib.suppress(ValueError):
                move = board.parse_san(raw)
        if move is None or move not in board.legal_moves:
            print(f'Illegal or unrecognised move: "{raw}".  Type "moves" for options.')
            continue

        board.push(move)
        history.append(board.fen())
        show_board(board)

    # ── Game over ─────────────────────────────────────────────────
    print(DIVIDER)
    result = board.result()
    if board.is_checkmate():
        winner = "White (Model)" if board.turn == chess.BLACK else "Black (You)"
        print(f"Checkmate! {winner} wins.  ({result})")
    elif board.is_stalemate():
        print(f"Stalemate — draw.  ({result})")
    elif board.is_insufficient_material():
        print(f"Insufficient material — draw.  ({result})")
    elif board.is_fifty_moves():
        print(f"Fifty-move rule — draw.  ({result})")
    elif board.is_repetition():
        print(f"Threefold repetition — draw.  ({result})")
    else:
        print(f"Game over: {result}")
    print(DIVIDER)


if __name__ == "__main__":
    play(parse_args())
