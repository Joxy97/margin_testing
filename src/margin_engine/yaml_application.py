"""Safe YAML configuration boundary for the complete margin application."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from data_manager import DataManagerConfig, PartitionedPickleDataStore
from download_manager import DownloadManagerConfig, LocalFirstProviderSelection
from download_unit import (
    DateChunker,
    InstrumentChunker,
    LocalCSVDataProvider,
    ProductChunker,
    YfinanceDataProvider,
)
from margin_calculator import (
    BQMMarginCalculatorConfig,
    BatchBQMExecutionPolicy,
    GreedyMarginCalculatorConfig,
    SequentialBQMExecutionPolicy,
)
from margin_calculator.optimization.optimization_solver.bqm_solver import (
    BQMSolverConfig,
)
from portfolio import Portfolio
from risk_state_generator import (
    CorrelatedReturnsVolaGridRiskStateGeneratorConfig,
    PCAGridProvider,
    PortfolioRiskStateBQMManager,
    ReturnsVolaGridRiskStateGeneratorConfig,
    StructuralQUBOTemplateCache,
)

from .config import MarginEngineConfig
from .margin_engine import MarginEngine
from .margin_report import MarginReport


@dataclass(frozen=True)
class MarginApplicationConfig:
    """Everything required to construct and run one margin calculation."""

    engine: MarginEngineConfig
    portfolio: Portfolio
    marginDate: date

    @classmethod
    def fromYaml(cls, path: str | Path) -> MarginApplicationConfig:
        """Load a complete application configuration from a safe YAML file."""
        config_path = Path(path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
        return _YamlConfigParser(config_path.parent).parse(document)

    def createEngine(self) -> MarginEngine:
        return MarginEngine(self.engine)

    def generateReport(self) -> MarginReport:
        """Construct the engine and run the configured margin calculation."""
        return self.createEngine().generateReport(self.portfolio, self.marginDate)


class _YamlConfigParser:
    """Translate primitive YAML values into typed application configuration."""

    def __init__(self, baseDirectory: Path) -> None:
        self.baseDirectory = baseDirectory

    def parse(self, document: Any) -> MarginApplicationConfig:
        root = self._mapping(document, "root")
        self._only(root, {"engine", "portfolio", "marginDate"}, "root")
        return MarginApplicationConfig(
            engine=self._engine(self._required(root, "engine", "root")),
            portfolio=self._portfolio(
                self._required(root, "portfolio", "root")
            ),
            marginDate=self._date(
                self._required(root, "marginDate", "root"),
                "marginDate",
            ),
        )

    def _engine(self, value: Any) -> MarginEngineConfig:
        config = self._mapping(value, "engine")
        self._only(
            config,
            {
                "downloadManager",
                "dataManager",
                "riskStateGenerator",
                "marginCalculator",
            },
            "engine",
        )
        return MarginEngineConfig(
            downloadManager=self._downloadManager(
                config.get("downloadManager", {})
            ),
            dataManager=self._dataManager(config.get("dataManager", {})),
            riskStateGenerator=self._riskStateGenerator(
                config.get("riskStateGenerator", {})
            ),
            marginCalculator=self._marginCalculator(
                config.get("marginCalculator", {})
            ),
        )

    def _downloadManager(self, value: Any) -> DownloadManagerConfig:
        config = self._mapping(value, "engine.downloadManager")
        self._only(
            config,
            {
                "providers",
                "providerSelection",
                "downloadAlgorithm",
                "downloadParameters",
                "requestParameters",
            },
            "engine.downloadManager",
        )
        providers = {
            str(name): self._provider(provider, f"providers.{name}")
            for name, provider in self._mapping(
                config.get("providers", {}),
                "engine.downloadManager.providers",
            ).items()
        }
        selection = config.get("providerSelection", "local_first")
        if selection != "local_first":
            raise ValueError(f"Unknown provider selection: {selection!r}")
        download_parameters = dict(
            self._mapping(
                config.get("downloadParameters", {}),
                "engine.downloadManager.downloadParameters",
            )
        )
        if "chunker" in download_parameters:
            download_parameters["chunker"] = self._chunker(
                download_parameters["chunker"],
                "engine.downloadManager.downloadParameters.chunker",
            )
        request_parameters = dict(
            self._mapping(
                config.get("requestParameters", {}),
                "engine.downloadManager.requestParameters",
            )
        )
        if "location" in request_parameters:
            request_parameters["location"] = str(
                self._path(request_parameters["location"])
            )
        return DownloadManagerConfig(
            providers=providers,
            providerSelection=LocalFirstProviderSelection(),
            downloadAlgorithm=str(
                config.get("downloadAlgorithm", "single_request")
            ),
            downloadParameters=download_parameters,
            requestParameters=request_parameters,
        )

    def _provider(self, value: Any, path: str) -> Any:
        if isinstance(value, str):
            provider_type = value
        else:
            config = self._mapping(value, path)
            self._only(config, {"type"}, path)
            provider_type = self._required(config, "type", path)
        providers = {
            "local_csv": LocalCSVDataProvider,
            "yfinance": YfinanceDataProvider,
        }
        try:
            return providers[str(provider_type)]()
        except KeyError as error:
            raise ValueError(f"Unknown data provider: {provider_type!r}") from error

    def _chunker(self, value: Any, path: str) -> Any:
        config = self._mapping(value, path)
        chunker_type = str(self._required(config, "type", path))
        if chunker_type in {"date", "instrument"}:
            self._only(config, {"type", "batchSize"}, path)
            batch_size = int(self._required(config, "batchSize", path))
            return (
                DateChunker(batch_size)
                if chunker_type == "date"
                else InstrumentChunker(batch_size)
            )
        if chunker_type == "product":
            self._only(config, {"type", "first", "second"}, path)
            return ProductChunker(
                self._chunker(self._required(config, "first", path), f"{path}.first"),
                self._chunker(
                    self._required(config, "second", path),
                    f"{path}.second",
                ),
            )
        raise ValueError(f"Unknown chunker: {chunker_type!r}")

    def _dataManager(self, value: Any) -> DataManagerConfig:
        path = "engine.dataManager"
        config = self._mapping(value, path)
        self._only(
            config,
            {"cacheType", "memorySize", "maxMemoryBytes", "backingStore"},
            path,
        )
        backing_store = None
        if config.get("backingStore") is not None:
            store = self._mapping(config["backingStore"], f"{path}.backingStore")
            self._only(store, {"type", "directory"}, f"{path}.backingStore")
            store_type = self._required(store, "type", f"{path}.backingStore")
            if store_type != "partitioned_pickle":
                raise ValueError(f"Unknown data backing store: {store_type!r}")
            backing_store = PartitionedPickleDataStore(
                self._path(
                    self._required(
                        store,
                        "directory",
                        f"{path}.backingStore",
                    )
                )
            )
        return DataManagerConfig(
            cacheType=str(config.get("cacheType", "lru")),
            memorySize=int(config.get("memorySize", 16)),
            maxMemoryBytes=(
                None
                if config.get("maxMemoryBytes") is None
                else int(config["maxMemoryBytes"])
            ),
            backingStore=backing_store,
        )

    def _riskStateGenerator(self, value: Any) -> Any:
        path = "engine.riskStateGenerator"
        config = dict(self._mapping(value, path))
        generator_type = str(config.pop("type", "returns_vola_grid"))
        pca_provider = config.pop("pcaGridProvider", None)
        if pca_provider is not None:
            provider_config = self._mapping(pca_provider, f"{path}.pcaGridProvider")
            self._only(
                provider_config,
                {"cacheType", "memorySize"},
                f"{path}.pcaGridProvider",
            )
            config["pcaGridProvider"] = PCAGridProvider(
                cacheType=str(provider_config.get("cacheType", "lru")),
                memorySize=int(provider_config.get("memorySize", 128)),
            )
        if "scenariosPerComponents" in config:
            config["scenariosPerComponents"] = tuple(
                int(item) for item in config["scenariosPerComponents"]
            )
        classes = {
            "returns_vola_grid": ReturnsVolaGridRiskStateGeneratorConfig,
            "correlated_returns_vola_grid": (
                CorrelatedReturnsVolaGridRiskStateGeneratorConfig
            ),
        }
        try:
            config_class = classes[generator_type]
        except KeyError as error:
            raise ValueError(
                f"Unknown risk-state generator: {generator_type!r}"
            ) from error
        return self._construct(config_class, config, path)

    def _marginCalculator(self, value: Any) -> Any:
        path = "engine.marginCalculator"
        config = dict(self._mapping(value, path))
        calculator_type = str(config.pop("type", "bqm"))
        if calculator_type == "greedy":
            self._only(config, set(), path)
            return GreedyMarginCalculatorConfig()
        if calculator_type != "bqm":
            raise ValueError(f"Unknown margin calculator: {calculator_type!r}")

        solver = self._mapping(config.pop("solver", {}), f"{path}.solver")
        self._only(
            solver,
            {"type", "constructorParameters", "solverParameters"},
            f"{path}.solver",
        )
        constructor_parameters = dict(
            self._mapping(
                solver.get("constructorParameters", {}),
                f"{path}.solver.constructorParameters",
            )
        )
        if "libraryPath" in constructor_parameters:
            constructor_parameters["libraryPath"] = self._path(
                constructor_parameters["libraryPath"]
            )
        solver_config = BQMSolverConfig(
            solverType=str(solver.get("type", "simulated_annealing")),
            constructorParameters=constructor_parameters,
            solverParameters=dict(
                self._mapping(
                    solver.get("solverParameters", {}),
                    f"{path}.solver.solverParameters",
                )
            ),
        )
        policy = self._executionPolicy(
            config.pop("executionPolicy", {"type": "sequential"}),
            f"{path}.executionPolicy",
        )
        manager = None
        if "structuralCacheMemorySize" in config:
            manager = PortfolioRiskStateBQMManager(
                StructuralQUBOTemplateCache(
                    int(config.pop("structuralCacheMemorySize"))
                )
            )
        self._only(config, {"modelParameters"}, path)
        return BQMMarginCalculatorConfig(
            solver=solver_config,
            modelParameters=dict(
                self._mapping(
                    config.get("modelParameters", {}),
                    f"{path}.modelParameters",
                )
            ),
            bqmManager=manager,
            executionPolicy=policy,
        )

    def _executionPolicy(self, value: Any, path: str) -> Any:
        if isinstance(value, str):
            policy_type = value
            config: Mapping[str, Any] = {}
        else:
            mutable = dict(self._mapping(value, path))
            policy_type = str(mutable.pop("type", "sequential"))
            config = mutable
        if policy_type == "sequential":
            self._only(config, set(), path)
            return SequentialBQMExecutionPolicy()
        if policy_type == "batch":
            self._only(config, {"batchSize"}, path)
            return BatchBQMExecutionPolicy(int(config.get("batchSize", 4)))
        raise ValueError(f"Unknown BQM execution policy: {policy_type!r}")

    def _portfolio(self, value: Any) -> Portfolio:
        path = "portfolio"
        config = self._mapping(value, path)
        self._only(config, {"weights", "cash"}, path)
        weights = self._mapping(
            self._required(config, "weights", path),
            f"{path}.weights",
        )
        return Portfolio(
            weights={str(key): Decimal(str(amount)) for key, amount in weights.items()},
            cash=Decimal(str(config.get("cash", 0))),
        )

    def _path(self, value: Any) -> Path:
        path = Path(str(value)).expanduser()
        return path.resolve() if path.is_absolute() else (self.baseDirectory / path).resolve()

    @staticmethod
    def _date(value: Any, path: str) -> date:
        if isinstance(value, date):
            return value
        try:
            return date.fromisoformat(str(value))
        except ValueError as error:
            raise ValueError(f"{path} must be an ISO date") from error

    @staticmethod
    def _mapping(value: Any, path: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise TypeError(f"{path} must be a mapping")
        return value

    @staticmethod
    def _required(config: Mapping[str, Any], key: str, path: str) -> Any:
        if key not in config:
            raise ValueError(f"Missing required YAML key: {path}.{key}")
        return config[key]

    @staticmethod
    def _only(config: Mapping[str, Any], allowed: set[str], path: str) -> None:
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(f"Unknown YAML keys at {path}: {sorted(unknown)}")

    @staticmethod
    def _construct(configClass: type, values: Mapping[str, Any], path: str) -> Any:
        fields = set(configClass.__dataclass_fields__)
        unknown = set(values) - fields
        if unknown:
            raise ValueError(f"Unknown YAML keys at {path}: {sorted(unknown)}")
        try:
            return configClass(**values)
        except TypeError as error:
            raise ValueError(f"Invalid configuration at {path}: {error}") from error
