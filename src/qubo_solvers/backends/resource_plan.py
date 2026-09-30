"""One owner for ordered shard assignment and per-worker memory admission."""

from dataclasses import dataclass
from collections.abc import Sequence


@dataclass(frozen=True)
class BQMResourcePlan:
    """Admitted device work; host coefficient retention is accounted separately."""

    shards: tuple[tuple[int, int], ...]
    workerBytes: tuple[int, ...]
    hostBytes: int = 0

    @classmethod
    def create(cls, sizes: Sequence[int], workers: int, hostBytes: int = 0):
        if workers <= 0:
            raise ValueError("BQM solver batchParallelism must be positive")
        if not sizes:
            return cls((), (), hostBytes)
        workers = min(workers, len(sizes))
        width, remainder = divmod(len(sizes), workers)
        shards, memory, start = [], [], 0
        for worker in range(workers):
            stop = start + width + (worker < remainder)
            shards.append((start, stop))
            memory.append(sum(sizes[start:stop]))
            start = stop
        return cls(tuple(shards), tuple(memory), hostBytes)

    def fits(self, budget: int | None) -> bool:
        """One oversized problem per worker is admitted to ensure progress."""
        return budget is None or all(
            memory <= budget or stop - start == 1
            for (start, stop), memory in zip(self.shards, self.workerBytes)
        )
