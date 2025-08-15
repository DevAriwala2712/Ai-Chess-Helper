import chess.pgn
import chess
import numpy as np
import pickle

piece_map = {
    chess.PAWN: 1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.ROOK: 4,
    chess.QUEEN: 5,
    chess.KING: 6
}

def board_to_array(board):
    arr = np.zeros(64, dtype=int)
    for square, piece in board.piece_map().items():
        value = piece_map[piece.piece_type]
        if piece.color == chess.BLACK:
            value = -value
        arr[square] = value
    return arr

pgn_path = "chess_data.pgn"        # <--- change to your PGN filename
max_games = 2000                       # how many games to extract from (adjust based on speed)

dataset = []
with open(pgn_path) as pgn:
    for i in range(max_games):
        game = chess.pgn.read_game(pgn)
        if game is None:
            break

        board = game.board()
        for move in game.mainline_moves():
            board_vec = board_to_array(board)
            dataset.append({
                "x": board_vec,
                "y": move.uci()         # move label in format like "e2e4"
            })
            board.push(move)

        if (i+1) % 200 == 0:
            print(f"Processed {i+1} games... total samples = {len(dataset)}")

# Save dataset to file
with open("chess_dataset.pkl", "wb") as f:
    pickle.dump(dataset, f)

print("Done! Total positions collected:", len(dataset))
print("Saved dataset to chess_dataset.pkl")
