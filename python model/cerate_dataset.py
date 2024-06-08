import numpy as np
import json
import os


def evaluate(state):
    # Check rows and columns
    for i in range(3):
        if state[i] == state[i + 3] == state[i + 6] != 0:
            return state[i]
        if state[3 * i] == state[3 * i + 1] == state[3 * i + 2] != 0:
            return state[3 * i]

    # Check diagonals
    if state[0] == state[4] == state[8] != 0:
        return state[0]
    if state[2] == state[4] == state[6] != 0:
        return state[2]

    # Check for tie
    if 0 not in state:
        return 0

    return None


def minimax(state, depth, isMaximizing, alpha, beta):
    # Base case: If the game is over, return the score
    score = evaluate(state)
    if score is not None:
        return score

    # If it's the maximizing player's turn
    if isMaximizing:
        best_score = -9999
        for i in range(9):
            if state[i] == 0:
                state[i] = 1
                score = minimax(state, depth + 1, False, alpha, beta)
                state[i] = 0
                best_score = max(score, best_score)
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
        return best_score
    else:
        best_score = 9999
        for i in range(9):
            if state[i] == 0:
                state[i] = -1
                score = minimax(state, depth + 1, True, alpha, beta)
                state[i] = 0
                best_score = min(score, best_score)
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
        return best_score


def find_best_move(state):
    best_score = -9999
    best_move = None
    move = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(9):
        if state[i] == 0:
            state[i] = 1
            alpha = -9999
            beta = 9999
            score = minimax(state, 0, False, alpha, beta)
            state[i] = 0
            if score > best_score:
                best_score = score
                best_move = i
    move[best_move] = 1
    return np.array(move)


def convert_board(board):
    new_board = []
    for i in range(9):
        if board[i] == 1:
            new_board += [0, 1, 0]
        elif board[i] == -1:
            new_board += [0, 0, 1]
        else:
            new_board += [1, 0, 0]
    return np.array(new_board)


def gernerate_all_possible():
    boards = []
    for a in range(-1, 2):
        for b in range(-1, 2):
            for c in range(-1, 2):
                for d in range(-1, 2):
                    for e in range(-1, 2):
                        for f in range(-1, 2):
                            for g in range(-1, 2):
                                for h in range(-1, 2):
                                    for i in range(-1, 2):
                                        board = [a, b, c, d, e, f, g, h, i]
                                        if np.sum(board) == 0 and evaluate(board) is None:
                                            boards.append(board)
    return np.array(boards)


def generate_all_initial_moves():
    boards = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    next_moves = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
    ]

    return np.array(boards), np.array(next_moves)


def generate_random_boards(n):
    boards = []
    for _ in range(n):
        while True:
            board = np.random.randint(-1, 2, 9)
            if (np.sum(board) == 0 or np.sum(board) == -1) and evaluate(board) is None:
                boards.append(board)
                break
    return np.array(boards)


def generate_next_states(boards):
    next_boards = []
    for board in boards:
        next_boards.append(find_best_move(board.copy()))
    return np.array(next_boards)


def main(samples):
    if not os.path.exists('datasets'):
        os.makedirs('datasets')

    if os.path.exists(f'datasets/{samples}_samples.json'):
        print(f"Dataset with {samples} samples already exists")
        return

    # boards = gernerate_all_possible()
    # boards = np.concatenate((boards, generate_random_boards(samples - len(boards))))
    boards = generate_random_boards(samples)

    # Generate next states
    next_boards = generate_next_states(boards)

    a, b = generate_all_initial_moves()

    boards = np.concatenate((boards, a))

    next_boards = np.concatenate((next_boards, b))

    # Convert boards to the format used in the dataset
    converted_boards = np.array([convert_board(board) for board in boards])

    print("Created dataset with", samples, "samples")
    dataset = {
        "states": [board for board in converted_boards.tolist()],
        "next_states": [board for board in next_boards.tolist()]
    }

    with open(f'datasets/{samples}_samples.json', 'w') as f:
        json.dump(dataset, f)
