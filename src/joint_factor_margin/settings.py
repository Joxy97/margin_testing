"""Typed composition settings; all numerical defaults come from existing models."""

from dataclasses import dataclass, field
import math

from download_manager.config import DownloadManagerConfig
from data_manager.config import DataManagerConfig
from margin_calculator.optimization import FactorStressQUBOConfig, FactorStressRepairConfig
from margin_calculator.optimization.optimization_solver.bqm_solver import BQMSolverConfig
from margin_calculator.optimization.optimization_solver.bqm_solver import BQMExecutionPolicy
from risk_state_generator import PCAGridProviderConfig
from risk_state_generator.volatility_forecast import VolatilityConfig


@dataclass(frozen=True)
class EstimationConfig:
    ew_window: int = 125
    ew_lambda: float = .93
    components: int = 2
    pcaGridProvider: PCAGridProviderConfig = field(default_factory=PCAGridProviderConfig)

    def __post_init__(self):
        for name in ("ew_window", "components"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"estimation.{name} must be an integer")
            if value < (2 if name == "ew_window" else 1):
                raise ValueError(f"estimation.{name} is too small")
        if isinstance(self.ew_lambda, bool) or not isinstance(self.ew_lambda, (int, float)):
            raise TypeError("estimation.ew_lambda must be numeric")
        if not math.isfinite(self.ew_lambda) or not 0 < self.ew_lambda <= 1:
            raise ValueError("estimation.ew_lambda must be in (0, 1]")
        if not isinstance(self.pcaGridProvider, PCAGridProviderConfig):
            raise TypeError("estimation.pcaGridProvider must be a PCAGridProviderConfig")


@dataclass(frozen=True)
class SearchConfig:
    type: str = "single_anchor"
    repairEnabled: bool = True
    repair: FactorStressRepairConfig = field(default_factory=FactorStressRepairConfig)

    def __post_init__(self):
        if self.type not in {"single_anchor", "boundary_aware"}:
            raise ValueError("search.type must be single_anchor or boundary_aware")
        if type(self.repairEnabled) is not bool:
            raise TypeError("search.repairEnabled must be boolean")
        if not isinstance(self.repair, FactorStressRepairConfig):
            raise TypeError("search.repair must be FactorStressRepairConfig")
        if self.type == "boundary_aware" and not self.repairEnabled:
            raise ValueError("boundary_aware search always repairs candidates")


@dataclass(frozen=True)
class DiagnosticsConfig:
    references: bool = False
    bounds: bool = False
    boundCoordinates: bool = False
    boundsBackend: str = "cpu"
    cudaDevice: int = 0

    def __post_init__(self):
        for name in ("references", "bounds", "boundCoordinates"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"diagnostics.{name} must be boolean")
        if self.boundsBackend not in {"cpu", "cuda"}:
            raise ValueError("diagnostics.boundsBackend must be cpu or cuda")
        if isinstance(self.cudaDevice, bool) or not isinstance(self.cudaDevice, int):
            raise TypeError("diagnostics.cudaDevice must be an integer")
        if self.cudaDevice < 0:
            raise ValueError("diagnostics.cudaDevice must be nonnegative")


@dataclass(frozen=True)
class EngineConfig:
    downloadManager: DownloadManagerConfig = field(default_factory=DownloadManagerConfig)
    dataManager: DataManagerConfig = field(default_factory=DataManagerConfig)
    estimation: EstimationConfig = field(default_factory=EstimationConfig)
    encoding: FactorStressQUBOConfig = field(default_factory=FactorStressQUBOConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    solver: BQMSolverConfig = field(default_factory=BQMSolverConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    volatility: VolatilityConfig | None = None
    executionPolicy: BQMExecutionPolicy | None = None

    def __post_init__(self):
        for name, cls in (("downloadManager", DownloadManagerConfig), ("dataManager", DataManagerConfig),
                          ("estimation", EstimationConfig), ("encoding", FactorStressQUBOConfig),
                          ("search", SearchConfig), ("solver", BQMSolverConfig),
                          ("diagnostics", DiagnosticsConfig)):
            if not isinstance(getattr(self, name), cls):
                raise TypeError(f"engine.{name} must be {cls.__name__}")
        if self.volatility is not None and not isinstance(self.volatility, VolatilityConfig):
            raise TypeError("engine.volatility must be VolatilityConfig or None")
        if self.executionPolicy is not None and not isinstance(self.executionPolicy, BQMExecutionPolicy):
            raise TypeError("engine.executionPolicy must be BQMExecutionPolicy or None")
        if (self.search.repairEnabled and self.estimation.components+1 > 3
                and self.search.repair.neighborhood == "full"):
            raise ValueError("more than three coordinates requires explicit pairwise repair")
