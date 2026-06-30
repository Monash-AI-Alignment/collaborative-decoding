from w2s_research.research_loop.stop_checker import _StopChecker, StopReason


def test_consecutive_error_stop():
    sc = _StopChecker(max_runtime=10_000, max_consecutive_errors=3)
    assert sc.check() is None
    sc.record_error()
    sc.record_error()
    assert sc.check() is None                 # 2 < 3
    sc.record_error()
    assert sc.check() is StopReason.MAX_ERRORS


def test_success_resets_error_streak():
    sc = _StopChecker(max_runtime=10_000, max_consecutive_errors=3)
    sc.record_error()
    sc.record_success()                       # reset
    sc.record_error()
    assert sc.check() is None


def test_timeout_stop():
    sc = _StopChecker(max_runtime=-1, max_consecutive_errors=3)   # already past
    assert sc.check() is StopReason.TIMEOUT
