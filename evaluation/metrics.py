"""Task metrics with explicit handling of malformed outputs."""


def sudoku_metrics(hyps, refs):
    if len(hyps) != len(refs) or not hyps:
        raise ValueError("nonempty hypotheses and references must align")

    def cells(text):
        return text.replace("|", "")

    def well_formed(grid):
        return len(grid) == 81 and all(c in "123456789" for c in grid)

    def valid(grid):
        if not well_formed(grid):
            return False
        rows = [grid[i:i + 9] for i in range(0, 81, 9)]
        cols = ["".join(row[i] for row in rows) for i in range(9)]
        boxes = [
            "".join(rows[r][c] for r in range(br, br + 3) for c in range(bc, bc + 3))
            for br in (0, 3, 6) for bc in (0, 3, 6)
        ]
        return all(sorted(group) == list("123456789") for group in rows + cols + boxes)

    predictions = [cells(h) for h in hyps]
    references = [cells(r) for r in refs]
    if not all(well_formed(r) for r in references):
        raise ValueError("Sudoku references must contain 81 digits from 1 to 9")
    # Missing positions count as wrong; longer predictions are also reported malformed.
    correct = sum(sum(a == b for a, b in zip(h[:81], r)) for h, r in zip(predictions, references))
    return {
        "n": len(hyps),
        "solved": sum(h == r for h, r in zip(hyps, refs)),
        "cell_accuracy": correct / (81 * len(hyps)),
        "valid": sum(valid(h) for h in predictions),
        "malformed": sum(not well_formed(h) for h in predictions),
    }
