"""Inclusive acquired-interval bookkeeping shared by distinct data schemas."""

from datetime import date, timedelta
from download_unit import DataRequest


class IntervalCoverage(dict[str, list[tuple[date, date]]]):
    """Track acquired calendar intervals, including those with no observations."""

    def add(self, command: DataRequest) -> None:
        for instrument in command.instruments:
            self[instrument] = self._mergeIntervals([
                *self.get(instrument, []), (command.start_date, command.end_date)])

    def missingRequests(self, command: DataRequest) -> list[DataRequest]:
        grouped = {}
        for instrument in command.instruments:
            for interval in self._missingIntervals(command.start_date, command.end_date,
                                                   self.get(instrument, [])):
                grouped.setdefault(interval, []).append(instrument)
        return [command.withChanges(instruments=tuple(instruments), start_date=start, end_date=end)
                for (start, end), instruments in sorted(grouped.items())]

    @staticmethod
    def _mergeIntervals(
        intervals: list[tuple[date, date]],
    ) -> list[tuple[date, date]]:
        merged: list[tuple[date, date]] = []
        for start, end in sorted(intervals):
            if not merged or (start - merged[-1][1]).days > 1:
                merged.append((start, end))
            else:
                previous_start, previous_end = merged[-1]
                merged[-1] = (previous_start, max(previous_end, end))
        return merged

    @staticmethod
    def _missingIntervals(
        start: date,
        end: date,
        coveredIntervals: list[tuple[date, date]],
    ) -> list[tuple[date, date]]:
        missing = []
        cursor = start
        for covered_start, covered_end in IntervalCoverage._mergeIntervals(
            coveredIntervals
        ):
            if covered_end < cursor:
                continue
            if covered_start > end:
                break
            if covered_start > cursor:
                missing.append(
                    (cursor, min(end, covered_start - timedelta(days=1)))
                )
            if covered_end >= end:
                return missing
            cursor = max(cursor, covered_end + timedelta(days=1))
            if cursor > end:
                break
        if cursor <= end:
            missing.append((cursor, end))
        return missing
