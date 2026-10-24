"""Connect existing scenario mathematics and valuation implementations."""

from dataclasses import dataclass, replace
from datetime import date
from functools import singledispatch
from typing import Protocol

import numpy as np
import pandas as pd

from option_pricing import EuropeanOptionBook, OptionBook
from risk_state_generator import (
    FactorStressModel, EuropeanOptionFactorStressModel, OptionFactorStressModel,
    ReturnsPCAKey, PCAGridProvider,
)
from risk_state_generator.volatility_forecast import (
    VolatilityConfig, VolatilityForecast, forecastVolatility, rescaleStressModel,
)
from .settings import EstimationConfig
from .values import CalculationRequest

StressModel = FactorStressModel | EuropeanOptionFactorStressModel


@singledispatch
def valuationModel(options, grid, portfolio, localPnlGradient=None):
    raise TypeError(f"unsupported valuation book: {type(options).__name__}")


@valuationModel.register(type(None))
def _(options, grid, portfolio, localPnlGradient=None):
    return FactorStressModel.fromPCAGrid(grid, portfolio)


@valuationModel.register(EuropeanOptionBook)
def _(options, grid, portfolio, localPnlGradient=None):
    return EuropeanOptionFactorStressModel.fromPCAGrid(grid, portfolio, options)


@valuationModel.register(OptionBook)
def _(options, grid, portfolio, localPnlGradient=None):
    return OptionFactorStressModel.fromPCAGrid(grid, portfolio, options, localPnlGradient=localPnlGradient)


@singledispatch
def valuationDiagnostics(model, radius):
    raise TypeError(f"unsupported factor model: {type(model).__name__}")


@valuationDiagnostics.register(FactorStressModel)
def _(model, radius):
    return {"type": "stocks"}


@valuationDiagnostics.register(EuropeanOptionFactorStressModel)
def _(model, radius):
    return {"type": "european", "impliedVolatilities": model.options.impliedVolatilities,
            "valuationDate": model.options.valuationDate, "horizonDate": model.options.horizonDate}


@valuationDiagnostics.register(OptionFactorStressModel)
def _(model, radius):
    return {"type": "vanilla", "calibrations": model.options.calibrations,
            "valuationDate": model.options.valuationDate, "horizonDate": model.options.horizonDate,
            "residualAlignment": model.residualAlignment, "domain": model.domainDiagnostics(radius)}


@singledispatch
def equityGeometry(model):
    raise TypeError(f"unsupported factor model: {type(model).__name__}")


@equityGeometry.register(FactorStressModel)
def _(model):
    return model


@equityGeometry.register(EuropeanOptionFactorStressModel)
def _(model):
    return model.equityModel


@singledispatch
def withOverlay(model, ratio):
    raise TypeError(f"unsupported factor model: {type(model).__name__}")


@withOverlay.register(FactorStressModel)
def _(model, ratio):
    return rescaleStressModel(model, ratio)


@withOverlay.register(EuropeanOptionFactorStressModel)
def _(model, ratio):
    return replace(model, equityModel=rescaleStressModel(model.equityModel, ratio))


@dataclass(frozen=True)
class PreparedScenario:
    model: StressModel
    calibrationStart: date
    calibrationEnd: date
    overlay: VolatilityForecast | None = None


class ScenarioBuilder(Protocol):
    def build(self, request: CalculationRequest, marketData: pd.DataFrame) -> PreparedScenario: ...


class ExistingScenarioBuilder:
    """Fit existing PCA, valuation and residual geometry without changing their math."""

    def __init__(self, estimation: EstimationConfig, provider: PCAGridProvider,
                 volatility: VolatilityConfig | None = None):
        self.estimation, self.provider, self.volatility = estimation, provider, volatility

    def build(self, request: CalculationRequest, marketData: pd.DataFrame) -> PreparedScenario:
        if "date" in marketData.columns:
            days = pd.to_datetime(marketData["date"], errors="raise")
        else:
            days = pd.to_datetime(marketData.index, errors="raise")
        if not days.is_unique:
            raise ValueError("market-data dates must be unique")
        key = ReturnsPCAKey(request.portfolio.instruments, self.estimation.ew_window,
                            request.marginDate, self.estimation.ew_lambda, self.estimation.components)
        grid = self.provider.getOrCreate(key, marketData)
        model = valuationModel(request.options, grid, request.portfolio.toPortfolio(), request.localPnlGradient)
        overlay = None
        if self.volatility is not None:
            # Reuse the exact PCA window preparation (including existing alignment).
            prices = marketData.copy()
            prices.index = pd.DatetimeIndex(days)
            prices = prices.sort_index().loc[:, list(request.portfolio.instruments)].ffill().dropna()
            prices = prices.loc[prices.index < pd.Timestamp(request.marginDate)].iloc[-self.estimation.ew_window-1:]
            overlay = forecastVolatility(np.diff(np.log(prices.to_numpy(dtype=float)), axis=0),
                prices.index, request.marginDate, grid.logReturnMean, grid.logReturnScale, self.volatility)
            model = withOverlay(model, overlay.ratio)
        return PreparedScenario(model, grid.calibrationStartDate, grid.calibrationEndDate, overlay)
