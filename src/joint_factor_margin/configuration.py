"""Strict YAML composition, reusing existing portfolio and acquisition readers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import date
from pathlib import Path
from hashlib import sha256
import json

import numpy as np
import yaml

from margin_engine.yaml_application import MarginConfigParser
from margin_calculator.optimization import FactorStressQUBOConfig, FactorStressRepairConfig
from margin_calculator.optimization.optimization_solver.bqm_solver import (
    BQMSolverConfig, SequentialBQMExecutionPolicy, BatchBQMExecutionPolicy,
)
from option_pricing import EuropeanOptionBook, OptionBook
from risk_state_generator import PCAGridProviderConfig, PCABackendConfig
from risk_state_generator.volatility_forecast import VolatilityConfig
from .settings import EngineConfig, EstimationConfig, SearchConfig, DiagnosticsConfig
from .values import ExposurePortfolio, CalculationRequest, primitive
from .application import createApplication


class _SettingsLoader(yaml.SafeLoader):
    """Reject repeated keys rather than silently overriding an experiment setting."""

    def construct_mapping(self, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                repeated = key in result
            except TypeError as error:
                raise ValueError("YAML mapping keys must be scalar") from error
            if repeated:
                raise ValueError(f"duplicate YAML key: {key}")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def section(value, allowed, path):
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    unknown = set(value)-set(allowed)
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown, key=str)}")
    return dict(value)


def typed(cls, value, path):
    return cls(**section(value, (field.name for field in fields(cls) if field.init), path))


def day(value, path):
    if type(value) is date:
        return value
    if not isinstance(value, str):
        raise TypeError(f"{path} must be an ISO date")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{path} must be an ISO date") from error


def required(value, key, path):
    if key not in value:
        raise ValueError(f"missing required key: {path}.{key}")
    return value[key]


@dataclass(frozen=True)
class BacktestConfig:
    dates: tuple[date, ...]
    confidenceLevel: float = .998
    name: str = "portfolio"

    def __post_init__(self):
        dates = tuple(self.dates)
        if not dates or any(type(item) is not date for item in dates):
            raise ValueError("backtest.dates must contain dates")
        if len(set(dates)) != len(dates):
            raise ValueError("backtest.dates must be unique")
        object.__setattr__(self, "dates", tuple(sorted(dates)))
        if isinstance(self.confidenceLevel, bool) or not isinstance(self.confidenceLevel, (int, float)):
            raise TypeError("backtest.confidenceLevel must be numeric")
        if not np.isfinite(self.confidenceLevel) or not 0 < self.confidenceLevel < 1:
            raise ValueError("backtest.confidenceLevel must lie between zero and one")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("backtest.name must be nonempty")


@dataclass(frozen=True)
class ApplicationConfig:
    request: CalculationRequest
    engine: EngineConfig
    configurationIdentity: str
    inputFiles: tuple[Path, ...] = ()
    backtest: BacktestConfig | None = None
    sourceText: str | None = None

    @classmethod
    def fromYaml(cls, path: str | Path):
        path = Path(path).expanduser().resolve()
        return cls.fromYamlText(path.read_text(), path.parent)

    @classmethod
    def fromYamlText(cls, text: str, baseDirectory: str | Path):
        return cls.fromMapping(yaml.load(text, Loader=_SettingsLoader), baseDirectory, sourceText=text)

    @classmethod
    def fromMapping(cls, document, baseDirectory: str | Path = ".", *, sourceText=None):
        root = section(document, {"schemaVersion", "marginDate", "portfolio", "options", "localPnlGradient", "engine", "backtest"}, "application")
        if type(root.get("schemaVersion")) is not int or root["schemaVersion"] != 1:
            raise ValueError("schemaVersion must be 1")
        base = Path(baseDirectory).expanduser().resolve()
        reader = MarginConfigParser(base)
        portfolio_input = required(root, "portfolio", "application")
        portfolio = ExposurePortfolio.fromPortfolio(reader.parsePortfolio(portfolio_input))
        margin_date = day(required(root, "marginDate", "application"), "marginDate")
        engine = section(root.get("engine", {}), {field.name for field in fields(EngineConfig)}, "engine")
        if "downloadManager" in engine:
            engine["downloadManager"] = reader.parseDownloadManager(engine["downloadManager"])
        if "dataManager" in engine:
            engine["dataManager"] = reader.parseDataManager(engine["dataManager"])
        estimation = section(engine.get("estimation", {}), {f.name for f in fields(EstimationConfig)}, "engine.estimation")
        provider = section(estimation.get("pcaGridProvider", {}), {f.name for f in fields(PCAGridProviderConfig)}, "engine.estimation.pcaGridProvider")
        provider["backend"] = typed(PCABackendConfig, provider.get("backend", {}), "engine.estimation.pcaGridProvider.backend")
        estimation["pcaGridProvider"] = PCAGridProviderConfig(**provider)
        engine["estimation"] = EstimationConfig(**estimation)
        engine["encoding"] = typed(FactorStressQUBOConfig, engine.get("encoding", {}), "engine.encoding")
        search = section(engine.get("search", {}), {f.name for f in fields(SearchConfig)}, "engine.search")
        search["repair"] = typed(FactorStressRepairConfig, search.get("repair", {}), "engine.search.repair")
        engine["search"] = SearchConfig(**search)
        solver = section(engine.get("solver", {}), {"type", "constructorParameters", "solverParameters"}, "engine.solver")
        solver_type = solver.get("type", "simulated_annealing")
        if not isinstance(solver_type, str) or not solver_type:
            raise TypeError("engine.solver.type must be a nonempty string")
        for name in ("constructorParameters", "solverParameters"):
            if name in solver and not isinstance(solver[name], Mapping):
                raise TypeError(f"engine.solver.{name} must be a mapping")
        constructor = dict(solver.get("constructorParameters", {}))
        if "libraryPath" in constructor:
            library = constructor["libraryPath"]
            if not isinstance(library, str) or not library:
                raise TypeError("engine.solver.constructorParameters.libraryPath must be a nonempty path")
            constructor["libraryPath"] = str((base/Path(library).expanduser()).resolve())
        engine["solver"] = BQMSolverConfig(solver_type, constructor, solver.get("solverParameters", {}))
        engine["diagnostics"] = typed(DiagnosticsConfig, engine.get("diagnostics", {}), "engine.diagnostics")
        if engine.get("volatility") is not None:
            engine["volatility"] = VolatilityConfig.fromMapping(engine["volatility"])
        if engine.get("executionPolicy") is not None:
            policy = section(engine["executionPolicy"], {"type", "batchSize", "maxBatchBytes", "memoryMultiplier", "prefetch"}, "engine.executionPolicy")
            kind = policy.pop("type", "sequential")
            if kind == "sequential":
                section(policy, (), "engine.executionPolicy")
                engine["executionPolicy"] = SequentialBQMExecutionPolicy()
            elif kind == "batch":
                engine["executionPolicy"] = BatchBQMExecutionPolicy(**policy)
            else:
                raise ValueError("engine.executionPolicy.type must be sequential or batch")
        engine = EngineConfig(**engine)
        options = None
        if "options" in root:
            options = _optionBook(root["options"], portfolio, reader)
        if engine.search.type == "boundary_aware" and not isinstance(options, OptionBook):
            raise ValueError("boundary_aware search requires options.type: vanilla")
        gradient = root.get("localPnlGradient")
        if gradient is not None:
            if not isinstance(gradient, Mapping) or set(gradient) != set(portfolio.instruments):
                raise ValueError("localPnlGradient must map every canonical portfolio instrument")
            gradient = np.asarray([gradient[instrument] for instrument in portfolio.instruments], dtype=float)
        request = CalculationRequest(portfolio, margin_date, options, gradient)
        backtest = None
        if "backtest" in root:
            value = section(root["backtest"], {"dates", "confidenceLevel", "name"}, "backtest")
            dates = required(value, "dates", "backtest")
            if not isinstance(dates, list):
                raise TypeError("backtest.dates must be a list")
            value["dates"] = tuple(day(item, "backtest.dates") for item in dates)
            backtest = BacktestConfig(**value)
            if options is not None:
                raise ValueError("option backtesting is unsupported: a dated realized-mark adapter is required")
        files = []
        parameters = engine.downloadManager.requestParameters
        for location in ([parameters["location"]] if "location" in parameters else []) + list(parameters.get("locations", ())):
            if "://" not in str(location):
                files.append(Path(location).resolve())
        if isinstance(portfolio_input, Mapping) and "csv" in portfolio_input:
            files.append((base/portfolio_input["csv"]).resolve())
        if engine.solver.solverType == "sbm":
            # Construction is lazy with respect to dlopen; include the selected
            # binary (also for default/environment paths) in experiment identity.
            files.append(engine.solver.createBQMSolver().libraryPath.expanduser().resolve())
        canonical = json.dumps({"document": primitive(root), "baseDirectory": str(base)}, sort_keys=True, allow_nan=False)
        return cls(request, engine, sha256(canonical.encode()).hexdigest(), tuple(sorted(set(files))), backtest, sourceText)

    def createApplication(self):
        return createApplication(self.engine, self.identity())

    def identity(self):
        """Include source, local data contents and dependency versions in resume identity."""
        from importlib.metadata import version, PackageNotFoundError
        digest = sha256(("joint-factor-margin-v1\0"+self.configurationIdentity).encode())
        for path in self.inputFiles:
            digest.update(str(path).encode())
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024*1024), b""):
                    digest.update(block)
        # Installed packages and editable checkouts both retain these Python sources.
        source = Path(__file__).resolve().parent.parent
        for name in ("joint_factor_margin", "margin_engine", "margin_calculator", "risk_state_generator",
                     "option_pricing", "portfolio", "numerics", "data_manager", "download_manager", "download_unit", "cache", "backtesting"):
            for path in sorted((source/name).rglob("*.py")):
                digest.update(str(path.relative_to(source)).encode())
                digest.update(path.read_bytes())
        for package in ("numpy", "pandas", "scipy", "dimod", "dwave-samplers", "torch", "cupy", "yfinance"):
            try:
                installed = version(package)
            except PackageNotFoundError:
                installed = "unavailable"
            digest.update(f"{package}={installed}".encode())
        return digest.hexdigest()


def _optionBook(value, portfolio, reader):
    config = section(value, {"type", "spotPrices", "positions", "marketPrices", "valuationDate", "horizonDate",
                             "riskFreeRate", "suppliedVolatilities", "volatilityBounds", "currency"}, "options")
    kind = config.pop("type", "european")
    if kind not in {"european", "vanilla"}:
        raise ValueError("options.type must be european or vanilla")
    spots = required(config, "spotPrices", "options")
    if not isinstance(spots, Mapping) or set(spots) != set(portfolio.instruments):
        raise ValueError("options.spotPrices must map every portfolio instrument to its dated spot")
    config["spotPrices"] = [spots[instrument] for instrument in portfolio.instruments]
    positions = required(config, "positions", "options")
    if not isinstance(positions, list):
        raise TypeError("options.positions must be a list")
    config["positions"] = tuple(reader.parseDerivativePosition(item, f"options.positions[{index}]")
                                for index, item in enumerate(positions))
    required(config, "marketPrices", "options")
    for name in ("valuationDate", "horizonDate"):
        config[name] = day(required(config, name, "options"), f"options.{name}")
    if kind == "european" and set(config) & {"suppliedVolatilities", "volatilityBounds", "currency"}:
        raise ValueError("suppliedVolatilities, volatilityBounds and currency require options.type: vanilla")
    cls = EuropeanOptionBook if kind == "european" else OptionBook
    return cls(instruments=portfolio.instruments, **config)
