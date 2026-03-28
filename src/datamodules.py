"""Chess dataset and datamodule for training from parquet files."""

import logging
import random

import bulletchess as bc
import pyarrow.parquet as pq
import torch as T
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.engine import encode_board

log = logging.getLogger(__name__)


class ChessStringDataset(Dataset):
    def __init__(self, parquet_path: str) -> None:
        self.table = pq.read_table(parquet_path, columns=["moves", "result"])
        self.results = self.table["result"].to_numpy()

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, idx: int) -> dict:
        moves = self.table["moves"][idx].as_py().split()
        result = self.results[idx]

        # Randomly choose a move up to the end of the game to progress to
        random_play = random.randint(1, len(moves))

        # Progress the board and encode
        board = bc.Board()
        for i in range(random_play):
            board.apply(bc.Move.from_uci(moves[i]))
        enc_board = encode_board(board)

        # Return as a dict for the model
        return {
            "board": T.from_numpy(enc_board),
            "result": T.tensor(result, dtype=T.long),
        }


class ChessDataModule(LightningDataModule):
    def __init__(
        self,
        *,
        train_path: str,
        test_path: str,
        num_workers: int = 4,
        batch_size: int = 256,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.train_set = ChessStringDataset(train_path)
        self.test_set = ChessStringDataset(test_path)
        log.info(
            f"Loaded full dataset with {len(self.train_set)} train samples "
            f"and {len(self.test_set)} test samples"
        )

    def _get_dataloader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.train_set, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.test_set)

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()

    def predict_dataloader(self) -> DataLoader:
        return self.val_dataloader()
