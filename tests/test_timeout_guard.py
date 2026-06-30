import time
import pytest
from w2s_research.core.timeout_guard import timeout


def test_timeout_raises_on_slow():
    with pytest.raises(TimeoutError):
        with timeout(1, "too slow"):
            time.sleep(3)


def test_timeout_no_raise_when_fast():
    with timeout(5):
        x = 1 + 1
    assert x == 2


def test_zero_seconds_is_noop():
    with timeout(0):          # disabled -> no alarm, body runs
        x = 7
    assert x == 7
