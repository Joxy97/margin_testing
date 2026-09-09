"""Own backtest identity, provenance, incremental resume and report publication."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from pathlib import Path
from collections.abc import Mapping
from types import MappingProxyType
import hashlib
import json

from .checkpoint import BacktestCheckpointStore
from .csv_reporter import BacktestCSVReporter, BacktestReportFiles
from .backtest_results import BacktestBatchResults
from .margin_backtester import MarginBacktester

if TYPE_CHECKING:
    from margin_engine import MarginApplicationConfig

NUMERICAL_MODEL_VERSION = 4


def experimentFingerprint(
    configBytes: bytes,
    application: MarginApplicationConfig,
) -> str:
    """Fingerprint canonical requests plus every configured local data file."""
    identity = f"margin-experiment-v{NUMERICAL_MODEL_VERSION}-timings-v2\0"
    digest = hashlib.sha256(identity.encode("ascii") + configBytes)
    for name, request in sorted(application.backtestRequests.items()):
        digest.update(str(name).encode("utf-8"))
        for instrument in request.portfolio.instruments:
            digest.update(str(instrument).encode("utf-8"))
            digest.update(
                str(request.portfolio.weights[instrument]).encode("utf-8")
            )
        digest.update(str(request.portfolio.cash).encode("utf-8"))
        for backtest_date in request.dates:
            digest.update(backtest_date.isoformat().encode("ascii"))

    request_parameters = application.engine.downloadManager.requestParameters
    configured_paths = []
    if request_parameters.get("location") is not None:
        configured_paths.append(request_parameters["location"])
    configured_paths.extend(request_parameters.get("locations", ()))
    for value in sorted(map(str, configured_paths)):
        path = Path(value).expanduser().resolve()
        digest.update(str(path).encode("utf-8"))
        if path.is_file():
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ExperimentOutcome:
    results: BacktestBatchResults
    reportFiles: Mapping[str, BacktestReportFiles] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "reportFiles", MappingProxyType(dict(self.reportFiles)))


class BacktestExperiment:
    def __init__(self, application: MarginApplicationConfig, checkpointStore=None,
                 outputDirectory: Path | None = None, configPath: Path | None = None,
                 configBytes: bytes | None = None):
        self.application = application
        self.checkpointStore = checkpointStore
        self.outputDirectory = outputDirectory
        self.configPath = configPath
        self.configBytes = configBytes if configBytes is not None else (
            None if configPath is None else configPath.read_bytes())

    @classmethod
    def fromYaml(cls, path: str | Path, outputDirectory: str | Path | None = None):
        from margin_engine import MarginApplicationConfig
        path = Path(path).expanduser().resolve()
        config_bytes = path.read_bytes()
        application = MarginApplicationConfig.fromYamlText(config_bytes.decode("utf-8"), path.parent)
        output = application.backtestOutputDirectory if outputDirectory is None else Path(outputDirectory).expanduser().resolve()
        if output is None:
            raise ValueError("set backtest.outputDirectory or provide outputDirectory")
        return cls(application, outputDirectory=output, configPath=path, configBytes=config_bytes)

    def run(self, resume: bool = False, onDayStarted=None) -> ExperimentOutcome:
        application = self.application
        if not application.backtestRequests:
            raise ValueError("YAML configuration does not contain a backtest block")
        store = self.checkpointStore
        if self.configPath is not None:
            config_bytes = self.configBytes
            fingerprint = experimentFingerprint(config_bytes, application)
            self.outputDirectory.mkdir(parents=True, exist_ok=True)
            (self.outputDirectory / "experiment_config.yaml").write_bytes(config_bytes)
            (self.outputDirectory / "experiment_manifest.json").write_text(json.dumps({
                "configPath": str(self.configPath),
                "configSha256": hashlib.sha256(config_bytes).hexdigest(),
                "experimentSha256": fingerprint, "checkpointSchema": 2,
                "measurementVersion": 2,
                "numericalModelVersion": NUMERICAL_MODEL_VERSION,
            }, indent=2, sort_keys=True))
            store = BacktestCheckpointStore(self.outputDirectory / ".checkpoints", fingerprint)
        if resume and store is None:
            raise ValueError("resume requires a checkpointStore")
        if store is not None and not resume:
            for name in application.backtestRequests:
                store.startFresh(name)
        completed = {name: store.load(name) for name in application.backtestRequests} if resume else None
        results = MarginBacktester().backtestMany(
            application.createEngine(), application.backtestRequests,
            application.backtestConfidenceLevel, completed,
            onNewDay=None if store is None else store.saveDay,
            onDayStarted=onDayStarted,
        )
        files = {} if self.outputDirectory is None else BacktestCSVReporter().write(results, self.outputDirectory)
        return ExperimentOutcome(results, files)
