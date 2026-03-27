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
    # 13 = En passant square
    # 14 = Positive bool
    # 15 = Negative bool
}


def bitboard_to_numpy_mask(bb: int) -> np.ndarray:
    """Converts a 64-bit integer bitboard into a (64,) boolean NumPy mask."""
    as_bytes = np.array([bb], dtype="<u8").view(np.uint8)
    return np.unpackbits(as_bytes, bitorder="little").astype(bool)


def encode_board(board: bc.Board) -> np.ndarray:
    """Encode the board as a 69-length vector of integers.

    - 64 integers for each square
    - 1 for the side to move
    - 4 for castling rights
    """
    # Initialize the empty board encoding
    enc_board = np.zeros(69, dtype=np.int64)

    # Loop through each piece type and color, fill in the encoding
    for p in bc.PIECE_TYPES:
        for c in [bc.WHITE, bc.BLACK]:
            bb = board[c, p]
            if bb:
                mask = bitboard_to_numpy_mask(bb)
                val = BOARD_TO_INT[str(bc.Piece(c, p))]
                enc_board[:64][mask] = val

    # Also include the location of an en passant square if it exists
    if board.en_passant_square:
        ep_square = board.en_passant_square.bb()
        ep_mask = bitboard_to_numpy_mask(ep_square)
        enc_board[:64][ep_mask] = 13

    # Include extra contextual information about the game state
    castling_types = [
        bc.WHITE_KINGSIDE,
        bc.WHITE_QUEENSIDE,
        bc.BLACK_KINGSIDE,
        bc.BLACK_QUEENSIDE,
    ]
    enc_board[64] = 14 if board.turn == bc.WHITE else 15
    enc_board[65:] = [14 if r in board.castling_rights else 15 for r in castling_types]
    return enc_board


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
