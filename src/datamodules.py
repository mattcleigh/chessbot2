"""Chess dataset and datamodule for training from parquet files."""

import logging

import chess
import numpy as np
import pyarrow.parquet as pq
import torch as T
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)

PIECE_TO_INT = {
    ".": 0,
    "P": 1,
    "N": 2,
    "B": 3,
    "R": 4,
    "Q": 5,
    "K": 6,
    "p": 7,
    "n": 8,
    "b": 9,
    "r": 10,
    "q": 11,
    "k": 12,
    "tW": 13,  # White to move
    "tB": 14,  # Black to move
}


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode the board as a (1, 65) integer tensor matching training input."""
    enc_board = np.zeros(65, dtype=np.int64)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)
            enc_board[(7 - row) * 8 + col] = PIECE_TO_INT[piece.symbol()]
    turn = "tW" if board.turn == chess.WHITE else "tB"
    enc_board[64] = PIECE_TO_INT[turn]
    return enc_board


class ChessStringDataset(Dataset):
    def __init__(self, parquet_path: str) -> None:
        table = pq.read_table(parquet_path, columns=["moves", "result"])
        self.moves = table["moves"].to_pylist()
        self.results = table["result"].to_pylist()
        self.rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.moves)

    def __getitem__(self, idx: int) -> dict:
        all_moves = self.moves[idx].split()
        max_idx = max(0, len(all_moves) - 2)  # Don't select AFTER checkmate
        random_ply = self.rng.integers(0, max_idx + 1)

        # Progress the board and encode
        board = chess.Board()
        for i in range(random_ply):
            board.push_san(all_moves[i])
        enc_board = encode_board(board)

        return {
            "board": T.from_numpy(enc_board),
            "result": T.tensor(self.results[idx], dtype=T.long),
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
