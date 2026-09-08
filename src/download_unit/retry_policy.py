"""Validated, declarative download retry and pacing settings."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class DownloadRetryPolicy:
    time: float = 1.0
    maxAttempts: int = 3
    maxDelay: float = 60.0
    jitter: float = 0.0
    requestInterval: float = 0.0

    def __post_init__(self) -> None:
        for name in ("time", "maxDelay", "jitter", "requestInterval"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if isinstance(self.maxAttempts, bool) or not isinstance(self.maxAttempts, int):
            raise TypeError("maxAttempts must be an integer")
        if self.maxAttempts < 1:
            raise ValueError("maxAttempts must be positive")
