"""Basic training script."""

import logging

import matplotlib.pyplot as plt
import pyarrow.parquet as pa
import rootutils

root = rootutils.setup_root(search_from=__file__, pythonpath=True)


log = logging.getLogger(__name__)
cfg_path = str(root / "configs")


def main() -> None:
    train_path = root / "data" / "train.parquet"
    labels = pa.read_table(train_path, columns=["result"])["result"].to_pylist()

    plt.hist(labels, bins=[-0.5, 0.5, 1.5, 2.5], rwidth=0.8)
    plt.xticks([0, 1, 2], ["White Win", "Black Win", "Draw"])
    plt.title("Distribution of Game Results in Training Data")
    plt.xlabel("Game Result")
    plt.ylabel("Count")
    plt.savefig(root / "results_distribution.png")
    plt.close()


if __name__ == "__main__":
    main()
