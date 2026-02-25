"""Shared constants and MajorityVoter for gesture recognition."""

import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import List

# LinkerHand O6 poses: [thumb_flex, thumb_abd, index, middle, ring, little]
POSES_O6 = {
    "paper":    [255, 255, 255, 255, 255, 255],
    "rock":     [70,  255, 0,   0,   0,   0],
    "scissors": [120, 70,  255, 255, 0,   0],
    "none":     [128, 128, 128, 128, 128, 128],
}

# Display colors (BGR)
GESTURE_COLORS = {
    "rock":     (0, 0, 255),
    "paper":    (0, 255, 0),
    "scissors": (255, 0, 0),
    "none":     (128, 128, 128),
}

# Battle mode: human gesture -> robot winning gesture
BATTLE_MAP = {
    "rock":     "paper",
    "paper":    "scissors",
    "scissors": "rock",
}


@dataclass
class MajorityVoter:
    """Sliding-window majority vote for stable gesture output.

    Supports two low-confidence strategies:
      - ``"none"``: push ``"none"`` for low-confidence frames.
      - ``"skip"``: discard low-confidence frames; clear stale window
        after *stale_timeout* seconds of no valid predictions.
    """

    window_size: int = 10
    conf_threshold: float = 0.6
    vote_mode: str = "none"
    stale_timeout: float = 1.0

    # Internal state (not exposed as init args)
    _window: deque = field(default_factory=deque, repr=False)
    _last_valid_time: float = field(default_factory=time.perf_counter, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self):
        self._window = deque(maxlen=self.window_size)
        self._last_valid_time = time.perf_counter()
        self._lock = threading.Lock()

    def push(self, gesture: str, confidence: float, now: float | None = None) -> None:
        """Push one prediction into the vote window."""
        if now is None:
            now = time.perf_counter()

        with self._lock:
            if confidence >= self.conf_threshold:
                self._window.append(gesture)
                self._last_valid_time = now
            elif self.vote_mode == "none":
                self._window.append("none")
            else:
                # skip mode: clear stale window
                if self._window and (now - self._last_valid_time) > self.stale_timeout:
                    self._window.clear()

    def majority(self) -> str:
        """Return current majority gesture (or ``"none"``)."""
        with self._lock:
            if not self._window:
                return "none"
            gesture, _count = Counter(self._window).most_common(1)[0]
            return gesture

    def clear(self) -> None:
        """Reset vote window."""
        with self._lock:
            self._window.clear()
            self._last_valid_time = time.perf_counter()
