 AI Helper — Chess AI with GUI
A lightweight chess-playing AI with a clickable GUI, built using Python-Chess, PyTorch, and Tkinter. It includes a full pipeline: parse PGNs, build a dataset, train a neural network to predict moves, and play via GUI or terminal.

Features
Tkinter GUI with Unicode chess pieces and clickable moves.

AI move prediction with a trained PyTorch model (ChessNet).

Board encoding: 64-length vector with signed integers for piece types.

Two play modes:

GUI assistant: click to input opponent moves; AI responds automatically.

Terminal play: play against the AI with UCI moves.

Data pipeline to convert PGN games into a supervised dataset for training.

Project Structure
ai_helper_gui.py: Tkinter GUI to visualize the board and play against the AI. Loads model and label encoder, predicts AI moves, and falls back to a random legal move if needed.

play_vs_ai.py: Terminal-based play loop with board printing and AI predictions.

train_model.py: End-to-end training script:

Loads chess_dataset.pkl

Encodes UCI moves with LabelEncoder

4-layer feedforward ChessNet over 64-dim board vectors

Saves chess_model.pth and label_encoder.pkl

make_dataset.py: Builds chess_dataset.pkl from a PGN file by iterating through game mainlines, storing (board_vector, next_move_uci).

convert_board.py: Utility to convert a python-chess Board to a 64-length numpy array.

parse_pgn.py: Quick PGN explorer to step through a few games.

test_play.py: Sanity-check the trained model’s move suggestion from the initial position.

Installation
Python 3.9+ recommended

Install dependencies:

pip install python-chess torch scikit-learn numpy

Data Preparation
Put a PGN at chess_data.pgn (or update the path in make_dataset.py).

Generate dataset:

python make_dataset.py
This creates chess_dataset.pkl.

Optional: Inspect PGN

python parse_pgn.py

Training
python train_model.py
Outputs:

chess_model.pth

label_encoder.pkl

Play — GUI
python ai_helper_gui.py
Flow:

Choose side in terminal (w/b).

Click source square, then destination.

AI replies; status bar shows actions.

Play — Terminal
python play_vs_ai.py

Enter UCI moves (e.g., e2e4). Type quit to exit.

How It Works
Board encoding: pieces mapped to integers:

Pawn=1, Knight=2, Bishop=3, Rook=4, Queen=5, King=6; white positive, black negative.

Model: 4-layer MLP (64→256→256→256→num_classes) trained with cross-entropy on next-move prediction.

Inference: Board vector → logits → argmax → UCI move via label encoder.

Robustness: If predicted move is illegal, a random legal move is used to continue play.

Notes and Tips
Model quality depends on PGN quality and size; increase max_games in make_dataset.py for better results.

Ensure label_encoder.pkl and chess_model.pth are in the working directory before running play scripts.

