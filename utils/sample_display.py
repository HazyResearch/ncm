"""Plain terminal rendering for text and Sudoku samples."""

import textwrap


def sudoku_board(ids):
    """Keep generated cells as-is, including blanks and malformed predictions."""
    ids = list(ids)
    while ids and ids[0] == 11:
        ids.pop(0)
    while ids and ids[-1] == 11:
        ids.pop()
    if len(ids) == 89:
        cells = [ids[row * 10 + col] for row in range(9) for col in range(9)]
        malformed = any(ids[row * 10 + 9] != 10 for row in range(8))
    elif len(ids) == 81:
        cells, malformed = ids, False
    else:
        return f"[Malformed board: {len(ids)} tokens]\n{ids}"
    lines = ['┌───────┬───────┬───────┐']
    for row in range(9):
        values = cells[row * 9:(row + 1) * 9]
        values = ['·' if value == 0 else str(value) if 1 <= value <= 9 else '?' for value in values]
        lines.append('│ ' + ' │ '.join(' '.join(values[col:col + 3]) for col in (0, 3, 6)) + ' │')
        if row in (2, 5):
            lines.append('├───────┼───────┼───────┤')
    lines.append('└───────┴───────┴───────┘')
    if malformed:
        lines.append('[Invalid row separator]')
    return '\n'.join(lines)


def text_block(label, text, width):
    paragraphs = [textwrap.fill(part, width=width, break_long_words=False,
                               break_on_hyphens=False) for part in text.strip().splitlines()]
    return label + '\n' + ('\n'.join(paragraphs) or '[Empty output]')


def format_sample(sample, number, width=96):
    header = f"Sample {number} · {sample['task']} · seed {sample['seed']}"
    if sample.get('dataset_index') is not None:
        header += f" · input {sample['dataset_index']}"
    sections = [header, '─' * min(width, len(header))]
    sudoku = sample['task'].startswith('sudoku')
    for name, label in [('prompt', 'Prompt'), ('output', 'Generated'), ('reference', 'Reference')]:
        if name not in sample:
            continue
        if sudoku:
            key = 'raw_ids' if name == 'output' else name + '_ids'
            sections.append(label + '\n' + sudoku_board(sample[key]))
        else:
            sections.append(text_block(label, sample[name], width))
    if sample.get('prompt_truncated'):
        sections.append('(Prompt shown is the model input, truncated to its context limit.)')
    if sample.get('trajectory'):
        sections.append('Trajectory · each hierarchy prefix decoded at full output length')
        for step in sample['trajectory']:
            sections.append('─' * width)
            label = f"Level {step['level']}/{len(sample['trajectory'])} · {step['codes']} codes"
            sections.append(label + '\n' + sudoku_board(step['raw_ids']) if sudoku
                            else text_block(label, step['text'], width))
    return '\n\n'.join(sections)
