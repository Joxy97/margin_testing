"""Calculation-local, overlap-aware host measurements and iterator cleanup."""

from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar

from .margin_report import MarginEngineTimings

T = TypeVar("T")


class CalculationMeasurements:
    def __init__(self, clock: Callable[[], float]) -> None:
        self.clock = clock
        self.started = clock()
        self.seconds = {"dataAcquisitionSeconds": 0.0, "marginCalculationSeconds": 0.0,
                        "riskStateGenerationSeconds": 0.0}

    def measure(self, stage: str, operation: Callable[[], T]) -> T:
        start = self.clock()
        try:
            return operation()
        finally:
            self.seconds[stage] += self.clock() - start

    def iterate(self, values: Iterable[T]) -> Iterator[T]:
        iterator = iter(values)
        try:
            while True:
                try:
                    value = self.measure("riskStateGenerationSeconds", lambda: next(iterator))
                except StopIteration:
                    return
                yield value
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()

    def timings(self) -> MarginEngineTimings:
        return MarginEngineTimings(**self.seconds, totalSeconds=self.clock() - self.started)
