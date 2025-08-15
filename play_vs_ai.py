import pickle
import torch
import chess
import numpy as np
from train_model import ChessNet

# ----- Helper: convert board to array -----
piece_map = {
    chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3,
    chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6
}
def board_to_array(board):
    arr = np.zeros(64, dtype=int)
    for sq, piece in board.piece_map().items():
        val = piece_map[piece.piece_type]
        if piece.color == chess.BLACK:
            val = -val
        arr[sq] = val
    return arr

# ----- Printable board with labels -----
def print_board_with_labels(board):
    board_str = board.unicode().split('\n')
    ranks = list(range(8, 0, -1))
    files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    for r, row in zip(ranks, board_str):
        print(r, row)
    print("  ", " ".join(files))

# ----- Load encoder and model -----
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)
num_classes = len(le.classes_)

model = ChessNet(num_classes)
state_dict = torch.load("chess_model.pth", map_location=torch.device("cpu"))
model.load_state_dict(state_dict)
model.eval()

# Predict next move
def predict_move(board):
    arr = board_to_array(board).astype(np.float32)
    x = torch.tensor(arr).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        idx = torch.argmax(logits, dim=1).item()
        return le.inverse_transform([idx])[0]

# ----- Choose side -----
side = input("Play as (w/b): ").strip().lower()
while side not in ("w", "b"):
    side = input("Enter 'w' for white or 'b' for black: ").strip().lower()

board = chess.Board()
print_board_with_labels(board)

# ----- Game loop -----
while not board.is_game_over():
    # If it's user's turn
    if (board.turn == chess.WHITE and side == "w") or (board.turn == chess.BLACK and side == "b"):
        user_move = input("Your move: ")
        if user_move == "quit":
            break
        try:
            board.push_uci(user_move)
        except:
            print("Invalid move, try again.")
            continue
        print("\nAfter your move:")
        print_board_with_labels(board)
        if board.is_game_over():
            break
    else:
        # AI move
        ai = predict_move(board)
        try:
            board.push_uci(ai)
            print("AI plays:", ai)
        except:
            import random
            ai = random.choice([m.uci() for m in board.legal_moves])
            board.push_uci(ai)
            print("AI predicted illegal move. Played random:", ai)
        print("\nAfter AI move:")
        print_board_with_labels(board)
        print("-" * 40)

print("Game Over! Result:", board.result())
