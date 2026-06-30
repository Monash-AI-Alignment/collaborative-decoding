"""Pure stop conditions for the autonomous loop (no SDK deps; CPU-testable).

Bounds Claude usage: a hard walltime cap AND a consecutive-error cap so a
rate-limit (or any repeated session failure) can't spin a GPU for the whole
walltime — it stops after MAX_CONSECUTIVE_ERRORS failures in a row.
"""
import os
import time
from enum import Enum
from typing import Optional


class StopReason(Enum):
    TIMEOUT = "timeout"
    USER_INTERRUPT = "user_interrupt"
    MAX_ERRORS = "max_consecutive_errors"


class _StopChecker:
    def __init__(self, max_runtime: float, max_consecutive_errors: int = None):
        self.max_runtime = max_runtime
        self.start_time = time.time()
        self.consecutive_errors = 0
        self.max_consecutive_errors = max_consecutive_errors or int(
            os.getenv("MAX_CONSECUTIVE_ERRORS", "4"))

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time

    def check(self) -> Optional[StopReason]:
        if self.elapsed_time >= self.max_runtime:
            return StopReason.TIMEOUT
        if self.consecutive_errors >= self.max_consecutive_errors:
            return StopReason.MAX_ERRORS
        return None

    def record_success(self):
        self.consecutive_errors = 0

    def record_error(self):
        self.consecutive_errors += 1
