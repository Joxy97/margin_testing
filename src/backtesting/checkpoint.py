"""Atomic per-day checkpoints for resumable backtests."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict
from datetime import date
from pathlib import Path

from .backtest_results import DailyBacktestResult, DailyBacktestTimings


class BacktestCheckpointStore:
    """Persist completed days, scoped to an exact experiment fingerprint."""

    def __init__(self, directory: str | Path, experimentFingerprint: str) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self.experimentFingerprint = str(experimentFingerprint)
        if not self.experimentFingerprint:
            raise ValueError("experimentFingerprint must not be empty")

    @staticmethod
    def _safeName(name: str) -> str:
        original = str(name)
        normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", original).strip("._")
        digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:12]
        return f"{normalized or 'portfolio'}-{digest}"

    def load(self, name: str) -> tuple[DailyBacktestResult, ...]:
        """Load matching completed days; ignore checkpoints from other runs."""
        path = self.directory / f"{self._safeName(name)}.json"
        days = {}
        if path.is_file():
            document = json.loads(path.read_text(encoding="utf-8"))
            if document.get("experimentFingerprint") == self.experimentFingerprint:
                days.update((day.date, day) for day in map(self._decodeDaily, document.get("days", ())))
        for path in sorted(self._dayDirectory(name).glob("*.json")):
            document = json.loads(path.read_text(encoding="utf-8"))
            if document.get("schema") != 2 or document.get("experimentFingerprint") != self.experimentFingerprint:
                raise ValueError(f"Incompatible checkpoint: {path}")
            day = self._decodeDaily(document["day"])
            if path.stem != day.date.isoformat():
                raise ValueError(f"Checkpoint date mismatch: {path}")
            days[day.date] = day
        return tuple(days[key] for key in sorted(days))

    def _dayDirectory(self, name: str) -> Path:
        fingerprint = hashlib.sha256(self.experimentFingerprint.encode()).hexdigest()
        return self.directory / "v2" / fingerprint / self._safeName(name)

    def startFresh(self, name: str) -> None:
        """Discard this experiment's prior completions before a new run."""
        for path in self._dayDirectory(name).glob("*.json"):
            path.unlink()
        legacy = self.directory / f"{self._safeName(name)}.json"
        if legacy.is_file():
            document = json.loads(legacy.read_text(encoding="utf-8"))
            if document.get("experimentFingerprint") == self.experimentFingerprint:
                legacy.unlink()

    def saveDay(self, name: str, result: DailyBacktestResult) -> None:
        """Atomically persist exactly one new completion; partial files stay invisible."""
        directory = self._dayDirectory(name)
        directory.mkdir(parents=True, exist_ok=True)
        document = {"schema": 2, "experimentFingerprint": self.experimentFingerprint,
                    "day": self._encodeDaily(result)}
        descriptor, temporary = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(document, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, directory / f"{result.date.isoformat()}.json")
        finally:
            Path(temporary).unlink(missing_ok=True)

    def save(self, name: str, results: tuple[DailyBacktestResult, ...]) -> None:
        """Atomically replace one portfolio checkpoint."""
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{self._safeName(name)}.json"
        temporary = path.with_suffix(path.suffix + ".tmp")
        document = {
            "experimentFingerprint": self.experimentFingerprint,
            "days": [self._encodeDaily(item) for item in results],
        }
        temporary.write_text(
            json.dumps(document, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    @staticmethod
    def _encodeDaily(result: DailyBacktestResult) -> dict[str, object]:
        return {
            "date": result.date.isoformat(),
            "margin": result.margin,
            "realizedPnL": result.realizedPnL,
            "grossExposure": result.grossExposure,
            "marginPercent": result.marginPercent,
            "breach": result.breach,
            "timings": asdict(result.timings),
            "comparisonMargins": dict(result.comparisonMargins),
        }

    @staticmethod
    def _decodeDaily(value: dict[str, object]) -> DailyBacktestResult:
        return DailyBacktestResult(
            date=date.fromisoformat(str(value["date"])),
            margin=float(value["margin"]),
            realizedPnL=float(value["realizedPnL"]),
            grossExposure=float(value["grossExposure"]),
            marginPercent=float(value["marginPercent"]),
            breach=bool(value["breach"]),
            timings=DailyBacktestTimings(**dict(value.get("timings", {}))),
            comparisonMargins=dict(value.get("comparisonMargins", {})),
        )
