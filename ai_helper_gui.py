import tkinter as tk
import chess
import numpy as np
import pickle
import torch
from train_model import ChessNet

# ============ LOAD MODEL & ENCODER ============
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)
num_classes = len(le.classes_)
model = ChessNet(num_classes)
state_dict = torch.load("chess_model.pth", map_location=torch.device("cpu"))
model.load_state_dict(state_dict)
model.eval()

# ============ UTILS ============
piece_to_unicode = {
    chess.PAWN: {True: "♙", False: "♟︎"},
    chess.ROOK: {True: "♖", False: "♜"},
    chess.KNIGHT: {True: "♘", False: "♞"},
    chess.BISHOP: {True: "♗", False: "♝"},
    chess.QUEEN: {True: "♕", False: "♛"},
    chess.KING: {True: "♔", False: "♚"},
}


def board_to_array(board):
    mapping = {chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3, chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6}
    arr = np.zeros(64, dtype=int)
    for sq, p in board.piece_map().items():
        v = mapping[p.piece_type]
        if not p.color: v *= -1
        arr[sq] = v
    return arr.astype(np.float32)


def ai_move(board):
    x = torch.tensor(board_to_array(board)).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
    idx = torch.argmax(logits, 1).item()
    return le.inverse_transform([idx])[0]


# ============ GUI ============
class ChessGUI:
    def __init__(self):
        self.board = chess.Board()
        self.root = tk.Tk()
        self.root.title("Secret AI Helper")
        self.buttons = {}
        self.selected = None
        self.user_color = None

        self.status = tk.Label(self.root, text="Select your side (white/black) in terminal")
        self.status.grid(row=8, column=0, columnspan=8)

        # draw board
        for r in range(8):
            for c in range(8):
                b = tk.Button(self.root, width=4, height=2,
                              command=lambda r=r, c=c: self.on_click(r, c))
                b.grid(row=r, column=c)
                self.buttons[(r, c)] = b

        # choose side in terminal
        side = input("Play as white or black? (w/b): ").strip().lower()
        self.user_color = chess.WHITE if side == 'w' else chess.BLACK

        self.update_gui()
        self.root.mainloop()

    def on_click(self, r, c):
        square = chess.square(c, 7 - r)
        if self.selected is None:
            self.selected = square
            self.status.config(text=f"Selected {chess.square_name(square)}")
        else:
            move = chess.Move(self.selected, square)
            uci = move.uci()
            if uci in [m.uci() for m in self.board.legal_moves]:
                # push opponent's move
                self.board.push_uci(uci)
                self.update_gui()
                self.status.config(text=f"Opponent played {uci}")
                self.root.update()

                # now AI responds
                if not self.board.is_game_over():
                    m = ai_move(self.board)
                    try:
                        self.board.push_uci(m)
                    except:
                        import random
                        m = random.choice([mv.uci() for mv in self.board.legal_moves])
                        self.board.push_uci(m)
                    self.status.config(text=f"AI plays {m} - now enter opponent move")
                    self.update_gui()
            else:
                self.status.config(text="Illegal move, pick again.")

            self.selected = None

    def update_gui(self):
        for r in range(8):
            for c in range(8):
                sq = chess.square(c, 7 - r)
                piece = self.board.piece_at(sq)
                if piece:
                    sym = piece_to_unicode[piece.piece_type][piece.color]
                    self.buttons[(r, c)]['text'] = sym
                else:
                    self.buttons[(r, c)]['text'] = ""


if __name__ == "__main__":
    ChessGUI()
