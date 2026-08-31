"""Atomic per-day checkpoints for resumable backtests."""

from __future__ import annotations

import hashlib
import json
import os
import re
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
        if not path.is_file():
            return ()
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("experimentFingerprint") != self.experimentFingerprint:
            return ()
        return tuple(self._decodeDaily(item) for item in document.get("days", ()))

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
