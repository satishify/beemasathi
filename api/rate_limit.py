"""In-memory token-bucket rate limiting (per client key)."""

import threading
import time
from typing import Callable, Dict, Tuple


class TokenBucketLimiter:
    def __init__(
        self,
        rate_per_minute: int = 30,
        burst: int = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.rate = max(1, rate_per_minute) / 60.0
        self.capacity = float(burst if burst is not None else max(1, rate_per_minute))
        self.clock = clock
        self._buckets: Dict[str, Tuple[float, float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> Tuple[bool, int]:
        """Return (allowed, retry_after_seconds)."""
        now = self.clock()
        with self._lock:
            tokens, last = self._buckets.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            if tokens < 1:
                wait = int((1 - tokens) / self.rate) + 1
                self._buckets[key] = (tokens, now)
                return False, wait
            self._buckets[key] = (tokens - 1, now)
            return True, 0

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()
