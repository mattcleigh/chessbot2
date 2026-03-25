"""Basic training script."""

import logging
from multiprocessing import Pool
import chess
import matplotlib.pyplot as plt
import pyarrow.parquet as pa
import rootutils
from tqdm import tqdm

root = rootutils.setup_root(search_from=__file__, pythonpath=True)


log = logging.getLogger(__name__)
cfg_path = str(root / "configs")


def main() -> None:
    train_path = root / "data" / "train.parquet"

    print("Loading results...")
    labels = pa.read_table(train_path, columns=["result"])["result"].to_pylist()
    plt.hist(labels, bins=[-0.5, 0.5, 1.5, 2.5], rwidth=0.8)
    plt.xticks([0, 1, 2], ["White Win", "Black Win", "Draw"])
    plt.title("Distribution of Game Results in Training Data")
    plt.xlabel("Game Result")
    plt.ylabel("Count")
    plt.savefig(root / "results_distribution.png")
    plt.close()

    # Lets play all the games and check if they are valid
    print("Loading games...")
    games = pa.read_table(train_path, columns=["moves"])["moves"].to_pylist()

    def check_game(moves: str) -> int:
        board = chess.Board()
        for move in moves.split():
            try:
                board.push_san(move)
            except ValueError:
                log.exception(f"Invalid move {move} in game: {moves}")
                return 1
        return 0

    for game in tqdm(games):
        check_game(game)


if __name__ == "__main__":
    main()
