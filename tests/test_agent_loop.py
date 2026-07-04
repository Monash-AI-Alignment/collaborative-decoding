import pytest

from w2s_research.research_loop.agent import AgentResult, validate_session_result
from w2s_research.research_loop.stop_checker import _StopChecker, StopReason


def test_zero_tool_session_raises():
    # A rate-limited CLI exits "successfully" in ~2s with only a limit notice and
    # zero tool calls — job 58089426 spun ~4550 such no-op sessions. Must raise so
    # the stop checker's consecutive-error cap ends the loop.
    r = AgentResult(success=True, output={}, duration=1.5, iteration_count=0)
    with pytest.raises(RuntimeError, match="no work"):
        validate_session_result(r)


def test_failed_session_raises():
    r = AgentResult(success=False, output={}, duration=3.0, iteration_count=0, error="boom")
    with pytest.raises(RuntimeError, match="boom"):
        validate_session_result(r)


def test_working_session_passes():
    r = AgentResult(success=True, output={}, duration=435.5, iteration_count=30)
    validate_session_result(r)   # no raise


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
