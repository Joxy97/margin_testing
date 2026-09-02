"""Safe YAML configuration boundary for the complete margin application."""

from __future__ import annotations

from collections.abc import Mapping
import csv
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import yaml

from data_manager import (
    DataManagerConfig,
    DerivativeQuoteDataManagerConfig,
    PartitionedPickleDataStore,
)
from download_manager import DownloadManagerConfig, LocalFirstProviderSelection
from download_unit import (
    DateChunker,
    DerivativeCSVDataProvider,
    InstrumentChunker,
    LocalCSVDataProvider,
    ProductChunker,
    YfinanceDataProvider,
)
from margin_calculator import (
    BQMMarginCalculatorConfig,
    BatchBQMExecutionPolicy,
    GreedyMarginCalculatorConfig,
    StateAwareGreedyMarginCalculatorConfig,
    SequentialBQMExecutionPolicy,
)
from margin_calculator.optimization.optimization_solver.bqm_solver import (
    BQMSolverConfig,
)
from portfolio import (
    DerivativePosition,
    DerivativesPortfolio,
    EquityContract,
    EquityOptionContract,
    FuturesContract,
    FuturesOptionContract,
    Portfolio,
)
from risk_state_generator import (
    CorrelatedReturnsVolaGridRiskStateGeneratorConfig,
    OptionScenarioRiskStateGeneratorConfig,
    PCAGridProvider,
    PortfolioRiskStateBQMVisitor,
    ReturnsVolaGridRiskStateGeneratorConfig,
    StructuralQUBOTemplateCache,
)

from .config import MarginEngineConfig
from .margin_engine import MarginEngine
from .margin_report import MarginReport

if TYPE_CHECKING:
    from backtesting import BacktestBatchResults, PortfolioBacktestRequest


@dataclass(frozen=True)
class MarginApplicationConfig:
    """Everything required to construct and run one margin calculation."""

    engine: MarginEngineConfig
    portfolio: Portfolio | DerivativesPortfolio
    marginDate: date
    backtestRequests: Mapping[str, PortfolioBacktestRequest] = field(
        default_factory=dict
    )
    backtestConfidenceLevel: float = 0.998
    backtestOutputDirectory: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "backtestRequests",
            MappingProxyType(dict(self.backtestRequests)),
        )

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

    def generateBacktest(
        self,
        checkpointStore: Any | None = None,
        resume: bool = False,
    ) -> BacktestBatchResults:
        """Run every named portfolio request from the YAML backtest block."""
        from backtesting import MarginBacktester

        if not self.backtestRequests:
            raise ValueError("YAML configuration does not contain a backtest block")
        if resume and checkpointStore is None:
            raise ValueError("resume requires a checkpointStore")
        return MarginBacktester().backtestMany(
            self.createEngine(),
            self.backtestRequests,
            self.backtestConfidenceLevel,
            (
                {
                    name: checkpointStore.load(name)
                    for name in self.backtestRequests
                }
                if resume and checkpointStore is not None
                else None
            ),
            (
                checkpointStore.save
                if checkpointStore is not None
                else None
            ),
        )


class _YamlConfigParser:
    """Translate primitive YAML values into typed application configuration."""

    def __init__(self, baseDirectory: Path) -> None:
        self.baseDirectory = baseDirectory

    def parse(self, document: Any) -> MarginApplicationConfig:
        root = self._mapping(document, "root")
        self._only(
            root,
            {"engine", "portfolio", "marginDate", "backtest"},
            "root",
        )
        portfolio = self._portfolio(
            self._required(root, "portfolio", "root")
        )
        if (
            isinstance(portfolio, DerivativesPortfolio)
            and root.get("backtest") is not None
        ):
            raise ValueError("derivative portfolio backtesting is not supported yet")
        requests, confidence_level, output_directory = self._backtest(
            root.get("backtest"),
            portfolio,
        )
        return MarginApplicationConfig(
            engine=self._engine(self._required(root, "engine", "root")),
            portfolio=portfolio,
            marginDate=self._date(
                self._required(root, "marginDate", "root"),
                "marginDate",
            ),
            backtestRequests=requests,
            backtestConfidenceLevel=confidence_level,
            backtestOutputDirectory=output_directory,
        )

    def _backtest(
        self,
        value: Any,
        defaultPortfolio: Portfolio,
    ) -> tuple[Mapping[str, Any], float, Path | None]:
        if value is None:
            return {}, 0.998, None
        from backtesting import PortfolioBacktestRequest

        path = "backtest"
        config = self._mapping(value, path)
        self._only(
            config,
            {
                "dates",
                "datesFromCsv",
                "portfolios",
                "confidenceLevel",
                "outputDirectory",
            },
            path,
        )
        requests: dict[str, PortfolioBacktestRequest] = {}
        if "dates" in config and "datesFromCsv" in config:
            raise ValueError("backtest must not define both dates and datesFromCsv")
        if "dates" in config or "datesFromCsv" in config:
            dates = (
                self._dates(config["dates"], f"{path}.dates")
                if "dates" in config
                else self._datesFromCsv(
                    config["datesFromCsv"],
                    f"{path}.datesFromCsv",
                )
            )
            requests["default"] = PortfolioBacktestRequest(
                defaultPortfolio,
                dates,
            )
        portfolios = self._mapping(config.get("portfolios", {}), f"{path}.portfolios")
        for name, value in portfolios.items():
            request_path = f"{path}.portfolios.{name}"
            request = self._mapping(value, request_path)
            self._only(
                request,
                {"portfolio", "dates", "datesFromCsv"},
                request_path,
            )
            if "dates" in request and "datesFromCsv" in request:
                raise ValueError(
                    f"{request_path} must not define both dates and datesFromCsv"
                )
            if "dates" not in request and "datesFromCsv" not in request:
                raise ValueError(
                    f"Missing required YAML key: {request_path}.dates"
                )
            requests[str(name)] = PortfolioBacktestRequest(
                self._portfolio(
                    self._required(request, "portfolio", request_path)
                ),
                (
                    self._dates(request["dates"], f"{request_path}.dates")
                    if "dates" in request
                    else self._datesFromCsv(
                        request["datesFromCsv"],
                        f"{request_path}.datesFromCsv",
                    )
                ),
            )
        if not requests:
            raise ValueError("backtest must define dates or named portfolios")
        confidence_level = float(config.get("confidenceLevel", 0.998))
        if not 0.0 < confidence_level < 1.0:
            raise ValueError("backtest.confidenceLevel must be between zero and one")
        output_directory = (
            self._path(config["outputDirectory"])
            if "outputDirectory" in config
            else None
        )
        return requests, confidence_level, output_directory

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
        if "locations" in request_parameters:
            locations = request_parameters["locations"]
            if isinstance(locations, (str, bytes)) or not isinstance(
                locations, list
            ):
                raise TypeError(
                    "engine.downloadManager.requestParameters.locations "
                    "must be a list"
                )
            request_parameters["locations"] = tuple(
                str(self._path(location)) for location in locations
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
            "derivative_csv": DerivativeCSVDataProvider,
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

    def _dataManager(self, value: Any) -> Any:
        path = "engine.dataManager"
        config = self._mapping(value, path)
        self._only(
            config,
            {"type", "cacheType", "memorySize", "maxMemoryBytes", "backingStore"},
            path,
        )
        manager_type = str(config.get("type", "close_prices"))
        if manager_type == "derivative_quotes":
            if set(config) != {"type"}:
                raise ValueError(
                    "derivative_quotes data manager accepts only the type key"
                )
            return DerivativeQuoteDataManagerConfig()
        if manager_type != "close_prices":
            raise ValueError(f"Unknown data manager: {manager_type!r}")
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
        if "volatilityShifts" in config:
            config["volatilityShifts"] = tuple(
                float(item) for item in config["volatilityShifts"]
            )
        classes = {
            "option_scenarios": OptionScenarioRiskStateGeneratorConfig,
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
        if calculator_type == "state_aware_greedy":
            self._only(config, {"pnlAnchor"}, path)
            return StateAwareGreedyMarginCalculatorConfig(
                pnlAnchor=str(config.get("pnlAnchor", "market"))
            )
        if calculator_type == "greedy":
            self._only(config, set(), path)
            return GreedyMarginCalculatorConfig()
        if calculator_type != "bqm":
            raise ValueError(f"Unknown margin calculator: {calculator_type!r}")

        comparison = config.pop("comparison", None)
        comparison_pnl_anchor = None
        if comparison is not None:
            comparison_config = self._mapping(
                comparison,
                f"{path}.comparison",
            )
            self._only(
                comparison_config,
                {"type", "pnlAnchor"},
                f"{path}.comparison",
            )
            if str(comparison_config.get("type", "state_aware_greedy")) != (
                "state_aware_greedy"
            ):
                raise ValueError("BQM comparison type must be state_aware_greedy")
            comparison_pnl_anchor = str(
                comparison_config.get("pnlAnchor", "market")
            )

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
        visitor = None
        if "structuralCacheMemorySize" in config:
            visitor = PortfolioRiskStateBQMVisitor(
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
            bqmVisitor=visitor,
            executionPolicy=policy,
            comparisonPnlAnchor=comparison_pnl_anchor,
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
            self._only(
                config,
                {"batchSize", "maxBatchBytes", "memoryMultiplier"},
                path,
            )
            return BatchBQMExecutionPolicy(
                batchSize=int(config.get("batchSize", 4)),
                maxBatchBytes=(
                    None
                    if config.get("maxBatchBytes") is None
                    else int(config["maxBatchBytes"])
                ),
                memoryMultiplier=float(config.get("memoryMultiplier", 3.0)),
            )
        raise ValueError(f"Unknown BQM execution policy: {policy_type!r}")

    def _portfolio(self, value: Any) -> Portfolio | DerivativesPortfolio:
        path = "portfolio"
        config = self._mapping(value, path)
        self._only(
            config,
            {"weights", "csv", "clientId", "cash", "positions", "metadata"},
            path,
        )
        if "positions" in config:
            if "weights" in config or "csv" in config:
                raise ValueError("portfolio must not combine positions with weights or csv")
            positions = config["positions"]
            if isinstance(positions, (str, bytes)) or not isinstance(positions, list):
                raise TypeError("portfolio.positions must be a list")
            return DerivativesPortfolio(
                positions=tuple(
                    self._derivativePosition(item, f"{path}.positions[{index}]")
                    for index, item in enumerate(positions)
                ),
                cash=Decimal(str(config.get("cash", 0))),
                metadata={
                    str(key): str(item)
                    for key, item in self._mapping(
                        config.get("metadata", {}), f"{path}.metadata"
                    ).items()
                },
            )
        if "metadata" in config:
            raise ValueError("portfolio.metadata is only valid with positions")
        if "weights" in config and "csv" in config:
            raise ValueError("portfolio must not define both weights and csv")
        if "csv" in config:
            weights = self._portfolioWeightsFromCsv(
                config["csv"],
                config.get("clientId"),
            )
        else:
            weights = self._mapping(
                self._required(config, "weights", path),
                f"{path}.weights",
            )
        return Portfolio(
            weights={str(key): Decimal(str(amount)) for key, amount in weights.items()},
            cash=Decimal(str(config.get("cash", 0))),
        )

    def _derivativePosition(self, value: Any, path: str) -> DerivativePosition:
        config = self._mapping(value, path)
        instrument_type = str(self._required(config, "instrumentType", path))
        common_keys = {
            "instrumentType", "symbol", "quantity", "multiplier", "currency"
        }
        allowed = {
            "equity": common_keys,
            "future": common_keys | {"expirationDate"},
            "futures_option": common_keys
            | {"expirationDate", "strike", "optionType", "exerciseStyle"},
            "equity_option": common_keys
            | {
                "expirationDate", "strike", "optionType", "exerciseStyle",
                "dividendYield",
            },
        }
        if instrument_type not in allowed:
            raise ValueError(f"Unknown instrument type: {instrument_type!r}")
        self._only(config, allowed[instrument_type], path)
        common = {
            "symbol": str(self._required(config, "symbol", path)),
            "multiplier": Decimal(str(config.get("multiplier", 1))),
            "currency": str(config.get("currency", "USD")),
        }
        if instrument_type == "equity":
            contract = EquityContract(**common)
        elif instrument_type == "future":
            contract = FuturesContract(
                **common,
                expirationDate=self._date(
                    self._required(config, "expirationDate", path),
                    f"{path}.expirationDate",
                ),
            )
        elif instrument_type in {"futures_option", "equity_option"}:
            option = {
                **common,
                "expirationDate": self._date(
                    self._required(config, "expirationDate", path),
                    f"{path}.expirationDate",
                ),
                "strike": Decimal(str(self._required(config, "strike", path))),
                "optionType": str(self._required(config, "optionType", path)),
                "exerciseStyle": str(config.get("exerciseStyle", "E")),
            }
            contract = (
                FuturesOptionContract(**option)
                if instrument_type == "futures_option"
                else EquityOptionContract(
                    **option,
                    dividendYield=float(config.get("dividendYield", 0.0)),
                )
            )
        return DerivativePosition(
            contract=contract,
            quantity=Decimal(str(self._required(config, "quantity", path))),
        )

    def _portfolioWeightsFromCsv(
        self,
        value: Any,
        clientId: Any,
    ) -> Mapping[str, str]:
        csv_path = self._path(value)
        with csv_path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        required_columns = {"client_id", "ticker", "weight"}
        if not rows:
            raise ValueError(f"Portfolio CSV {csv_path} must contain data")
        if required_columns.issubset(rows[0]):
            return self._longPortfolioWeights(csv_path, rows, clientId)
        if "client_id" in rows[0]:
            return self._widePortfolioWeights(csv_path, rows, clientId)
        raise ValueError(
            f"Portfolio CSV {csv_path} must use long-form "
            "client_id/ticker/weight columns or a wide client_id row"
        )

    @staticmethod
    def _selectPortfolioRows(
        csvPath: Path,
        rows: list[dict[str, str]],
        clientId: Any,
    ) -> list[dict[str, str]]:
        available_clients = tuple(dict.fromkeys(row["client_id"] for row in rows))
        selected_client = (
            str(clientId)
            if clientId is not None
            else available_clients[0] if len(available_clients) == 1 else None
        )
        if selected_client is None:
            raise ValueError("portfolio.clientId is required for a multi-client CSV")
        selected_rows = [
            row for row in rows if row["client_id"] == selected_client
        ]
        if not selected_rows:
            raise ValueError(
                f"Portfolio CSV {csvPath} has no client {selected_client!r}"
            )
        return selected_rows

    def _longPortfolioWeights(
        self,
        csvPath: Path,
        rows: list[dict[str, str]],
        clientId: Any,
    ) -> Mapping[str, str]:
        selected_rows = self._selectPortfolioRows(csvPath, rows, clientId)
        weights: dict[str, Decimal] = {}
        for row in selected_rows:
            ticker = str(row["ticker"])
            weights[ticker] = weights.get(ticker, Decimal("0")) + Decimal(
                str(row["weight"])
            )
        return {ticker: str(weight) for ticker, weight in weights.items()}

    def _widePortfolioWeights(
        self,
        csvPath: Path,
        rows: list[dict[str, str]],
        clientId: Any,
    ) -> Mapping[str, str]:
        selected_rows = self._selectPortfolioRows(csvPath, rows, clientId)
        if len(selected_rows) != 1:
            raise ValueError(
                f"Wide portfolio CSV {csvPath} must have one row per client"
            )
        return {
            ticker: weight
            for ticker, weight in selected_rows[0].items()
            if ticker != "client_id" and weight not in {None, ""}
        }

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

    def _dates(self, value: Any, path: str) -> tuple[date, ...]:
        if isinstance(value, (str, bytes)) or not isinstance(value, list):
            raise TypeError(f"{path} must be a list of dates")
        return tuple(
            self._date(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )

    def _datesFromCsv(self, value: Any, path: str) -> tuple[date, ...]:
        config = self._mapping(value, path)
        self._only(config, {"path", "startDate", "endDate"}, path)
        csv_path = self._path(self._required(config, "path", path))
        start = (
            self._date(config["startDate"], f"{path}.startDate")
            if "startDate" in config
            else None
        )
        end = (
            self._date(config["endDate"], f"{path}.endDate")
            if "endDate" in config
            else None
        )
        with csv_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            date_columns = [
                name
                for name in (reader.fieldnames or ())
                if name.casefold() == "date"
            ]
            if len(date_columns) != 1:
                raise ValueError(f"Date CSV {csv_path} must contain a date column")
            date_column = date_columns[0]
            dates = tuple(self._date(row[date_column], path) for row in reader)
        return tuple(
            item
            for item in dates
            if (start is None or item >= start) and (end is None or item <= end)
        )

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
