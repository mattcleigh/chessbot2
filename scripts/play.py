"""Play against a trained chess model.

Usage:
    python scripts/play.py --ckpt path/to/checkpoint.ckpt [--top-n 5]
"""

import argparse
import contextlib
import sys
from functools import partial

import bulletchess as bc
import rootutils
import torch as T
from torch.optim import AdamW

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

from src.engine import Node, mcts_search, update_root
from src.lightning_utils import linear_warmup_cosine_decay
from src.modules import TransformerConfig
from src.networks import ChessModel

# Allow loading checkpoints that reference these.
T.serialization.add_safe_globals([
    TransformerConfig,
    partial,
    AdamW,
    linear_warmup_cosine_decay,
])

DIVIDER = "─" * 44


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play against the chess model.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="saved_models/chessbot2/scale_teeny_saved/checkpoints/last.ckpt",
        help="Path to a Lightning checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run the model on (cpu, cuda, auto).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=0,
        help="How many top moves to show in the evaluation (0 for all).",
    )
    return parser.parse_args()


def choose_device(device: str) -> T.device:
    if device == "auto":
        return T.device("cuda" if T.cuda.is_available() else "cpu")
    return T.device(device)


def load_model(ckpt_path: str, device: T.device) -> ChessModel:
    model = ChessModel.load_from_checkpoint(
        ckpt_path,
        map_location=device,
        weights_only=True,
    )
    # Uncompile the model
    if hasattr(model.model, "_orig_mod"):
        model.model = model.model._orig_mod  # noqa: SLF001
    model.eval()
    return model


def show_eval_table(scored, top_n: int) -> None:
    n = len(scored) if top_n == 0 else min(top_n, len(scored))
    print(DIVIDER)
    print(f"  {'Move':<8} {'Score':>6}")
    print(DIVIDER)
    for move, score in scored[:n]:
        bar_len = int(score * 10) + 10  # Scale score from [-1, 1] to [0, 20]
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f" {move!s:<8} {score:+.2f}  {bar}")
    if n < len(scored):
        print(f"  … and {len(scored) - n} more moves")
    print(DIVIDER)


def print_help() -> None:
    print("""
Commands:
  <move>     Enter a move in UCI (e.g. e7e5) or SAN (e.g. e5, Nf6).
  undo       Undo the last full move (your move + model's move).
  board      Redisplay the board.
  moves      Show all your legal moves.
  quit       Exit the game.
  help       Show this message.
""")


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    print(f"Using device: {device}")
    print(f"Loading checkpoint: {args.ckpt}")

    model = load_model(args.ckpt, device)
    board = bc.Board()
    top_n = args.top_n
    root = Node(board.copy())

    print("\n♟  Chess Bot - Interactive Play  ♟")
    print("Model plays as WHITE.  You play as BLACK.")
    print('Type "help" for a list of commands.\n')
    print(board.pretty())

    while True:
        # Model's turn (White)
        if board.turn == bc.WHITE:
            print("Model is thinking …")
            scored, root = mcts_search(model, root)
            show_eval_table(scored, top_n)
            move, score = scored[0]
            print(f"Model plays: {move}  Score: {score:+.2f} ")

        # Human's turn (Black)
        else:
            raw = input("Your move: ").strip()
            if not raw:
                continue

            # Handle non-move commands first
            cmd = raw.lower()
            if cmd in {"quit", "exit", "q"}:
                print("Thanks for playing!")
                sys.exit(0)
            if cmd == "help":
                print_help()
                continue
            if cmd == "board":
                print(board.pretty())
                continue
            if cmd == "moves":
                legal = sorted(str(m) for m in board.legal_moves())
                print("Legal moves:", ", ".join(legal))
                continue
            if cmd == "undo":
                if len(board.history) < 2:
                    print("Nothing to undo!")
                    continue
                board.undo()  # undo model's reply
                board.undo()  # undo player's move
                print("Undone last full move.")
                print(board.pretty())
                continue

            # Try understanding the user's move
            move = None
            with contextlib.suppress(ValueError):
                move = bc.Move.from_uci(raw)
            if move is None:
                with contextlib.suppress(ValueError):
                    move = bc.Move.from_san(raw, board)
            if move is None:
                with contextlib.suppress(ValueError):
                    move = board.legal_moves()[int(raw) - 1]
            if move is None or move not in board.legal_moves():
                print(
                    f'Illegal or unrecognised move: "{raw}".  Type "moves" for options.'
                )
                continue

        # Apply the move and show the board
        board.apply(move)
        root = update_root(root, move)
        print(board.pretty())

        # Check for game end conditions
        game_end_message = ""
        if board in bc.CHECKMATE:
            winner = "White (Model)" if board.turn == bc.BLACK else "Black (You)"
            game_end_message = f"Checkmate! {winner} wins."
        elif board in bc.DRAW:
            game_end_message = "It's a draw!"
        if game_end_message:
            print(DIVIDER)
            print(game_end_message)
            print(DIVIDER)
            break


if __name__ == "__main__":
    main()
