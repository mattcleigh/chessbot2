"""Chess dataset and datamodule for training from parquet files."""

import logging
import random

import bulletchess as bc
import numpy as np
import pyarrow.parquet as pq
import torch as T
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)

BOARD_TO_INT = {
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
    True: 14,  # Positive sign for context
    False: 15,  # Negative sign for context
}


def encode_board(board: bc.Board) -> np.ndarray:
    """Encode the board as a 69-length vector of integers.

    - 64 integers for each square
    - 1 for the side to move
    - 4 for castling rights
    """
    # Initialize the empty board encoding
    enc_board = np.zeros(69, dtype=np.int64)

    # Loop through the squares if it contains a piece, encode it
    for square in bc.SQUARES:
        piece = board[square]
        if piece:
            row, col = divmod(square.index(), 8)
            enc_board[(7 - row) * 8 + col] = BOARD_TO_INT[str(piece)]

    # Also include the location of an en passant square if it exists
    if board.en_passant_square:
        ep_square = board.en_passant_square.index()
        row, col = divmod(ep_square, 8)
        enc_board[(7 - row) * 8 + col] = BOARD_TO_INT["e"]

    # Include extra contextual information about the game state
    enc_board[64] = BOARD_TO_INT[board.turn == bc.WHITE]
    enc_board[65] = BOARD_TO_INT[bc.WHITE_KINGSIDE in board.castling_rights]
    enc_board[66] = BOARD_TO_INT[bc.WHITE_QUEENSIDE in board.castling_rights]
    enc_board[67] = BOARD_TO_INT[bc.BLACK_KINGSIDE in board.castling_rights]
    enc_board[68] = BOARD_TO_INT[bc.BLACK_QUEENSIDE in board.castling_rights]
    return enc_board


class ChessStringDataset(Dataset):
    def __init__(self, parquet_path: str) -> None:
        table = pq.read_table(parquet_path, columns=["moves", "result"])
        self.moves = table["moves"].to_pylist()
        self.results = table["result"].to_pylist()

    def __len__(self) -> int:
        return len(self.moves)

    def __getitem__(self, idx: int) -> dict:
        moves = self.moves[idx].split()
        result = self.results[idx]

        # Randomly choose a move up to the end of the game to progress to
        random_play = random.randint(1, len(moves))

        # Progress the board and encode
        board = bc.Board()
        for i in range(random_play):
            board.apply(bc.Move.from_san(moves[i], board))
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
