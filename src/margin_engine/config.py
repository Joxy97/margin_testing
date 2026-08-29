"""Typed top-level margin-engine configuration."""

from dataclasses import dataclass, field

from data_manager import DataManagerConfig
from download_manager import DownloadManagerConfig
from margin_calculator import BQMMarginCalculatorConfig, MarginCalculatorConfig
from risk_state_generator import (
    ReturnsVolaGridRiskStateGeneratorConfig,
    RiskStateGeneratorConfig,
)


@dataclass(frozen=True)
class MarginEngineConfig:
    """Complete dependency configuration for one margin engine."""

    downloadManager: DownloadManagerConfig = field(
        default_factory=DownloadManagerConfig
    )
    dataManager: DataManagerConfig = field(default_factory=DataManagerConfig)
    riskStateGenerator: RiskStateGeneratorConfig = field(
        default_factory=ReturnsVolaGridRiskStateGeneratorConfig
    )
    marginCalculator: MarginCalculatorConfig = field(
        default_factory=BQMMarginCalculatorConfig
    )
