# Step 1 — parse the PGN file and view what it contains
import chess.pgn

# path to your downloaded .pgn file
pgn_file_path = "chess_data.pgn"   # <-- change this to the actual filename

games_to_read = 3  # just read the first 3 games to understand what's inside

with open(pgn_file_path) as pgn:
    for n in range(games_to_read):
        game = chess.pgn.read_game(pgn)
        print(f"\n--------- GAME {n+1} ---------")
        print(game.headers)   # metadata like White, Black, Result, etc.

        board = game.board()
        for move in game.mainline_moves():
            board.push(move)
            print("Move played:", move)
            print(board)             # print ASCII board
            input("Press Enter for next move...")  # walk through manually
