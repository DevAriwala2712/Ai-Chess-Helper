import torch
import pickle
import chess
import numpy as np
from train_model import ChessNet  # reuse from earlier

import numpy as np
import chess

# --- paste this helper function ---
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
# --- end paste ---


# Load model
# ===== Load label encoder first =====
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)
num_classes = len(le.classes_)

# ===== Build model using that number =====
model = ChessNet(num_classes)
state_dict = torch.load("chess_model.pth", map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()


# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Function to get model's move suggestion
def predict_move(board):
    arr = board_to_array(board).astype(np.float32)
    x = torch.tensor(arr).unsqueeze(0)  # shape = (1,64)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        move_index = torch.argmax(probs, dim=1).item()
        move_uci = le.inverse_transform([move_index])[0]
        return move_uci

# Test from initial board
board = chess.Board()
print("Initial board:")
print(board)

move = predict_move(board)
print("\nModel recommends:", move)
