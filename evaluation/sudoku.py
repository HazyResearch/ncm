"""Reference-free verification for adaptive Sudoku generation."""


def _cells(ids):
    ids = list(ids)
    # Prompts may wrap the 89-token board in BOS delimiters (token 11).
    if ids and ids[0] == 11:
        ids = ids[1:]
    if ids and ids[-1] == 11:
        ids = ids[:-1]
    if len(ids) == 89:
        if any(ids[i] != 10 for i in range(9, 89, 10)):
            return None
        ids = [v for i, v in enumerate(ids) if i % 10 != 9]
    if len(ids) != 81 or any(v not in range(10) for v in ids):
        return None
    return ids


def is_sudoku_solution(candidate, puzzle):
    """Require all 27 Sudoku units and every nonzero input clue to match.

    Accept flattened 81-cell boards or SudokuTokenizer's row-separated boards,
    optionally wrapped in BOS tokens. Malformed inputs never trigger an exit.
    """
    board, clues = _cells(candidate), _cells(puzzle)
    if board is None or clues is None or 0 in board:
        return False
    if any(clue and clue != value for clue, value in zip(clues, board)):
        return False
    units = [board[i:i + 9] for i in range(0, 81, 9)]
    units += [board[i::9] for i in range(9)]
    units += [
        [board[(r + dr) * 9 + c + dc] for dr in range(3) for dc in range(3)]
        for r in (0, 3, 6) for c in (0, 3, 6)
    ]
    return all(set(unit) == set(range(1, 10)) for unit in units)
