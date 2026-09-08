"""Per-execution coefficient retention across producer and solve threads."""

from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class BQMExecutionMemory:
    retainedCoefficientBytes: int
    activeCoefficientBytes: int
    producerCoefficientBytes: int
    peakRetainedCoefficientBytes: int


class ExecutionMemoryTracker:
    """Account source arrays, including a producer's pending oversized item.

    This counts logical array retention, not RSS or solver workspace. An observer
    runs serially on either execution thread and must return promptly.
    """

    def __init__(self, observer):
        self.observer = observer
        self.retained = self.active = self.peak = 0
        self.lock = Lock()

    def update(self, retainedDelta=0, active=None, finished=False):
        with self.lock:
            self.retained += retainedDelta
            if active is not None:
                self.active = active
            if finished:
                self.retained = self.active = 0
            self.peak = max(self.peak, self.retained)
            if self.observer is not None:
                self.observer(BQMExecutionMemory(
                    self.retained, self.active, self.retained - self.active, self.peak))
