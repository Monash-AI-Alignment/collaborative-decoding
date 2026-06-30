"""Main-thread SIGALRM timeout guard (no-op off the main thread or when seconds<=0).

Used to fail-fast if a strong-model generation hangs (e.g. the vLLM engine core
died and the client blocks forever). vLLM/HF decode run on the main thread, so the
alarm fires; the judge's worker threads are unaffected (no-op there).
"""
import signal
import threading


class timeout:
    def __init__(self, seconds, message="operation timed out"):
        self.seconds = seconds
        self.message = message
        self._active = (bool(seconds and seconds > 0)
                        and threading.current_thread() is threading.main_thread())
        self._prev = None

    def _handler(self, signum, frame):
        raise TimeoutError(self.message)

    def __enter__(self):
        if self._active:
            self._prev = signal.signal(signal.SIGALRM, self._handler)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, *exc):
        if self._active:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self._prev)
        return False
