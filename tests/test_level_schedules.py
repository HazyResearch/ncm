import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.registry import level_schedulers  # noqa: E402


@pytest.mark.parametrize("name", sorted(set(level_schedulers) - {"log"}) + ["log"])
@pytest.mark.parametrize("n", [2, 3, 5, 8])
def test_level_schedule_is_strictly_increasing_between_endpoints(name, n):
    levels = level_schedulers[name](1, 64, n)
    assert len(levels) == n and levels[0] == 1 and levels[-1] == 64
    assert all(b > a for a, b in zip(levels, levels[1:])), levels
