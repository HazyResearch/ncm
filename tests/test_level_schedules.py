import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.registry import level_schedulers  # noqa: E402


@pytest.mark.parametrize("name", sorted(level_schedulers))
@pytest.mark.parametrize("n", [2, 3, 5, 8])
def test_level_schedule_is_strictly_increasing_between_endpoints(name, n):
    levels = level_schedulers[name](1, 64, n)
    assert len(levels) == n and levels[0] == 1 and levels[-1] == 64
    assert all(b > a for a, b in zip(levels, levels[1:])), levels


@pytest.mark.parametrize(
    "n, expected",
    [
        (8, [1, 2, 8, 17, 29, 45, 65, 89]),
        (12, [1, 2, 3, 7, 12, 19, 27, 36, 47, 59, 73, 89]),
    ],
)
def test_quadratic_schedule_pinned_values(n, expected):
    # Normalizing by (n-2)**2 instead of (n-1)**2 hit `end` one level early;
    # round_levels then produced a +1 plateau before the final jump
    # (n=8: [..., 62, 63, 89]), which is still strictly increasing.
    assert level_schedulers["quadratic"](1, 89, n) == expected
