import chess
import numpy as np

# piece to number mapping
piece_map = {
    chess.PAWN: 1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.ROOK: 4,
    chess.QUEEN: 5,
    chess.KING: 6
}

def board_to_array(board):
    """
    Converts a python-chess board to a 64-length numpy array.
    White pieces = positive, Black = negative, empty = 0.
    """
    arr = np.zeros(64, dtype=int)
    for square, piece in board.piece_map().items():
        value = piece_map[piece.piece_type]
        if piece.color == chess.BLACK:
            value = -value
        arr[square] = value
    return arr

# test
board = chess.Board()  # starting position
print(board)
vec = board_to_array(board)
print("\nVector representation:\n", vec)
