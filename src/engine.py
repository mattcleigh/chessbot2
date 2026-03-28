from __future__ import annotations

import math
import operator
from typing import TYPE_CHECKING

import bulletchess as bc
import numpy as np
import torch as T
from tqdm import trange

if TYPE_CHECKING:
    from src.networks import ChessModel

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


@T.no_grad()
def evaluate_board(
    model: ChessModel,
    board: bc.Board,
    color: bc.Color = bc.WHITE,
) -> float:
    """Evaluate the board and return a score from -1 to 1."""
    enc = encode_board(board)
    probs = model(T.from_numpy(enc).unsqueeze(0).to(model.device)).squeeze(0)
    score = (probs[0] - probs[1]).item()  # White win - Black win
    if color == bc.BLACK:
        score = -score
    return score


@T.no_grad()
def evaluate_moves(model: ChessModel, board: bc.Board):
    """Score every legal move and return a list sorted bestto worst for white."""
    scored = []
    for move in board.legal_moves():
        board.apply(move)
        if board in bc.CHECKMATE:
            score = 1.0 if board.turn == bc.BLACK else -1.0
        elif board in bc.STALEMATE or board in bc.INSUFFICIENT_MATERIAL:
            score = 0.0
        else:
            score = evaluate_board(model, board)
        scored.append((move, score))
        board.undo()

    # Sort by white White - Black win probability (descending).
    scored.sort(key=operator.itemgetter(1), reverse=True)
    return scored


class Node:
    """A node in the game tree, used for MCTS."""

    def __init__(self, board: bc.Board, parent: Node | None = None) -> None:
        self.board = board
        self.parent = parent
        self.children: list[Node] = []
        self.visits = 0
        self.value = 0.0
        self.value_sum = 0.0
        self.untried_moves = list(board.legal_moves())

    @property
    def q(self) -> float:
        """The average value of this node."""
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def ucb1(self, color: bc.Color, c: float = 1.41) -> float:
        """Calculate the UCB1 score for this node."""
        if self.visits == 0:
            return float("inf")
        exploit = self.q
        explore = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        if self.board.turn != color:
            return exploit + explore  # It if is now the opponents turn (normal)
        return exploit - explore  # If it is now your turn, invert

    def select_child(self, color: bc.Color) -> Node:
        """Select the child with the highest UCB1 score."""
        return max(self.children, key=lambda n: n.ucb1(color))


def mcts_search(
    model: ChessModel,
    root: Node,
    num_simulations: int = 1000,
    color: bc.Color = bc.WHITE,
) -> list[tuple[bc.Move, float, Node]]:
    """Perform MCTS to find the best move from the given board state."""
    for _ in trange(num_simulations, ncols=100):
        node = root  # Always start at the root for each simulation

        # Selection - move down the tree until we find a node with untried moves
        while not node.untried_moves and node.children:
            node = node.select_child(color)

        # Expansion - if we find untried moves, expand the tree to one of them
        if node.untried_moves:
            move = node.untried_moves.pop()
            new_board = node.board.copy()
            new_board.apply(move)
            child_node = Node(new_board, parent=node)
            node.children.append(child_node)
            node = child_node  # Move to that child for the next step

        # Simulation - Evaluate the board at this node using the model
        if node.board in bc.CHECKMATE:
            score = 1.0 if node.board.turn == bc.BLACK else -1.0
        elif node.board in bc.DRAW:
            score = 0.0
        else:
            score = evaluate_board(model, node.board, color)

        # Backpropagation - We need to update the score for all nodes in the path
        while node is not None:
            node.visits += 1
            node.value_sum += score
            node = node.parent

    # Sort children by visits descending, then extract the moves and scores
    sorted_children = sorted(root.children, key=lambda c: c.visits, reverse=True)
    return [(child.board.history[-1], child.q) for child in sorted_children], root


def update_root(root: Node, move: bc.Move) -> Node:
    """After a move is played, update the root to keep the subtree stats."""
    for child in root.children:
        if child.board.history[-1] == move:
            child.parent = None  # Detach from old parent
            return child
    # If we didn't find the move among the children, rare but possible, must be fresh
    new_board = root.board.copy()
    new_board.apply(move)
    return Node(new_board)
