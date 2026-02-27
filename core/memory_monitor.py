"""Periodic memory usage monitor with memory bar warnings."""

import gc
import time
from typing import Optional

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False


class MemoryMonitor:
    """RSS monitor — call tick() once per frame.

    After a warmup period (skip startup model loading), alerts when
    RSS grows by more than warn_delta_mb between checks.
    """

    def __init__(self, check_interval: float = 30.0,
                 warn_delta_mb: float = 10.0,
                 warmup: float = 120.0,
                 warn_duration: float = 5.0):
        self._check_interval = check_interval
        self._warn_delta = warn_delta_mb
        self._warmup = warmup            # seconds to skip after start
        self._warn_duration = warn_duration
        self._start_time = time.monotonic()
        self._last_check = 0.0
        self._last_rss: float = 0.0
        self._peak_rss: float = 0.0
        # Warning state
        self._warn_msg: Optional[str] = None
        self._warn_expire: float = 0.0

    @property
    def rss_mb(self) -> float:
        """Current RSS in MB (0 if not yet measured)."""
        return self._last_rss

    @property
    def peak_mb(self) -> float:
        """Peak RSS in MB."""
        return self._peak_rss

    @property
    def warning(self) -> bool:
        """True while memory growth alert is active."""
        return self._warn_msg is not None and time.monotonic() < self._warn_expire

    def collect(self) -> None:
        """Run garbage collection on demand (e.g. at demo switch)."""
        gc.collect()

    def tick(self) -> None:
        if not _PSUTIL:
            return
        now = time.monotonic()
        # Periodic RSS check (skip warmup)
        if now - self._last_check >= self._check_interval:
            self._check_rss(now)
            self._last_check = now

    def _check_rss(self, now: float) -> None:
        rss = psutil.Process().memory_info().rss / (1024 * 1024)
        self._peak_rss = max(self._peak_rss, rss)
        in_warmup = (now - self._start_time) < self._warmup
        if self._last_rss > 0 and not in_warmup:
            delta = rss - self._last_rss
            if delta > self._warn_delta:
                self._warn_msg = (
                    f"[MEMORY] {rss:.0f} MB (+{delta:.0f} MB in "
                    f"{self._check_interval:.0f}s)"
                )
                self._warn_expire = now + self._warn_duration
        self._last_rss = rss
