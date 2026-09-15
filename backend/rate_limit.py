"""In-memory sliding-window rate limiter for LLM endpoints on public deployments."""

import threading
import time
from collections import deque
from collections.abc import Callable

MAX_TRACKED_KEYS = 10_000


class SlidingWindowRateLimiter:
    """Allows at most ``limit`` events per key within a rolling window.

    State lives in process memory, which matches the single-worker server
    (``uvicorn --workers 1``).
    """

    def __init__(
        self, window_seconds: float = 60.0, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._window = window_seconds
        self._clock = clock
        self._events: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int) -> bool:
        now = self._clock()
        cutoff = now - self._window
        with self._lock:
            if len(self._events) > MAX_TRACKED_KEYS:
                self._drop_stale_keys(cutoff)
            events = self._events.setdefault(key, deque())
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= limit:
                return False
            events.append(now)
            return True

    def reset(self) -> None:
        with self._lock:
            self._events.clear()

    def _drop_stale_keys(self, cutoff: float) -> None:
        self._events = {k: v for k, v in self._events.items() if v and v[-1] > cutoff}
