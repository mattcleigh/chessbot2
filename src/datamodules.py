"""Chess dataset and datamodule for training from parquet files."""

import logging
import random

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
    "e": 13,  # En passant square
    "+": 14,  # Positive sign for context
    "-": 15,  # Negative sign for context
}


def encode_flag(flag: bool) -> int:
    return PIECE_TO_INT["+"] if flag else PIECE_TO_INT["-"]


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode the board as a 69-length vector of integers.

    - 64 integers for each square
    - 1 for the side to move
    - 4 for castling rights
    """
    enc_board = np.zeros(69, dtype=np.int64)
    ep_square = board.ep_square
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)
            enc_board[(7 - row) * 8 + col] = PIECE_TO_INT[piece.symbol()]
        elif square == ep_square:
            row, col = divmod(square, 8)
            enc_board[(7 - row) * 8 + col] = PIECE_TO_INT["e"]
    enc_board[64] = encode_flag(board.turn)
    enc_board[65] = encode_flag(board.has_kingside_castling_rights(chess.WHITE))
    enc_board[66] = encode_flag(board.has_queenside_castling_rights(chess.WHITE))
    enc_board[67] = encode_flag(board.has_kingside_castling_rights(chess.BLACK))
    enc_board[68] = encode_flag(board.has_queenside_castling_rights(chess.BLACK))
    return enc_board


class ChessStringDataset(Dataset):
    def __init__(self, parquet_path: str) -> None:
        table = pq.read_table(parquet_path, columns=["moves", "result"])
        self.moves = table["moves"].to_pylist()
        self.results = table["result"].to_pylist()

    def __len__(self) -> int:
        return len(self.moves)

    def __getitem__(self, idx: int) -> dict:
        all_moves = self.moves[idx].split()
        random_ply = random.randint(1, len(all_moves))
        random_ply = 0
        # Progress the board and encode
        board = chess.Board()
        for i in range(random_ply):
            board.push_san(all_moves[i])
        enc_board, enc_ctxt = encode_board(board)

        return {
            "board": T.from_numpy(enc_board),
            "context": T.from_numpy(enc_ctxt),
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
