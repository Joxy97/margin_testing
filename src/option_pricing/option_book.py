"""Dated European/American equity option book with explicit calibration status."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from .european_option_book import EuropeanOptionBook
from .ju_zhong import OptionPricingError, VanillaPriceContext


@dataclass(frozen=True)
class OptionCalibration:
    backend: str
    status: str
    volatility: float
    residual: float


@dataclass(frozen=True)
class OptionBook(EuropeanOptionBook):
    """Common equity/index calls and puts; fixed IV, continuous yield, one currency.

    suppliedVolatilities entries are explicit model inputs (None means calibrate).
    Their mark residual is retained. Calibrated American IV is a bracketed root,
    not a uniqueness assertion. All failures retain the position index.
    """

    suppliedVolatilities: tuple[float | None, ...] | None = None
    volatilityBounds: tuple[float, float] = (.01, 5.)
    currency: str = "USD"
    contexts: tuple[VanillaPriceContext, ...] = field(init=False)
    calibrations: tuple[OptionCalibration, ...] = field(init=False)

    def __post_init__(self) -> None:
        bounds = tuple(self.volatilityBounds)
        if len(bounds) != 2 or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in bounds):
            raise TypeError("volatilityBounds must contain two numbers")
        if not np.isfinite(bounds).all() or not 0 < bounds[0] < bounds[1]:
            raise ValueError("volatilityBounds must be finite, positive and increasing")
        object.__setattr__(self, "volatilityBounds", bounds)
        if not isinstance(self.currency, str) or not self.currency:
            raise ValueError("currency must be a nonempty string")
        if self.suppliedVolatilities is not None:
            supplied = tuple(self.suppliedVolatilities)
            if len(supplied) != len(self.positions):
                raise ValueError("suppliedVolatilities must match positions")
            for value in supplied:
                if value is not None:
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise TypeError("supplied volatility must be numeric or None")
                    if not np.isfinite(value) or value <= 0:
                        raise ValueError("supplied volatility must be positive and finite")
            object.__setattr__(self, "suppliedVolatilities", supplied)
        super().__post_init__()
        contexts, calibrations = [], []
        for index, volatility in enumerate(self.impliedVolatilities):
            try:
                initial = self._context(index, float(volatility), self.valuationDate)
                context = self._context(index, float(volatility), self.horizonDate)
                residual = float(initial.evaluate(self.spotPrices[self._underlyingIndices[index]]))-self.marketPrices[index]
            except ValueError as error:
                raise OptionPricingError(f"positions[{index}] preparation: {error}") from error
            supplied = self.suppliedVolatilities is not None and self.suppliedVolatilities[index] is not None
            status = "supplied" if supplied else ("bracketed_root" if self.positions[index].contract.exerciseStyle == "A" else "calibrated")
            calibrations.append(OptionCalibration(initial.backend, status, float(volatility), float(residual)))
            contexts.append(context)
        object.__setattr__(self, "contexts", tuple(contexts))
        object.__setattr__(self, "calibrations", tuple(calibrations))

    def _validateContract(self, index, contract) -> None:
        if contract.currency != self.currency:
            raise ValueError(f"positions[{index}] currency differs from book; an FX layer is required")
        if contract.exerciseStyle == "A" and self.riskFreeRate < 0:
            raise OptionPricingError(f"positions[{index}] Ju-Zhong requires nonnegative rate")

    def _context(self, index, volatility, when) -> VanillaPriceContext:
        contract = self.positions[index].contract
        return VanillaPriceContext(float(contract.strike), (contract.expirationDate-when).days/365.,
            self.riskFreeRate, contract.dividendYield, volatility, contract.optionType, contract.exerciseStyle)

    def _calibrate(self, index: int, underlying: int) -> float:
        supplied = self.suppliedVolatilities is not None and self.suppliedVolatilities[index] is not None
        contract = self.positions[index].contract
        spot, mark, strike = self.spotPrices[underlying], self.marketPrices[index], float(contract.strike)
        sign = 1 if contract.optionType == "C" else -1
        tolerance = 1e-10*max(1., mark)
        if contract.exerciseStyle == "A":
            intrinsic = max(sign*(spot-strike), 0.)
            time = (contract.expirationDate-self.valuationDate).days/365.
            rate, yield_ = self.riskFreeRate, contract.dividendYield
            times = [0., time]
            if rate > 0 and yield_ > 0 and rate != yield_:
                stationary = np.log(yield_*spot/(rate*strike))/(yield_-rate)
                if 0 < stationary < time:
                    times.append(stationary)
            deterministic = max(0., *(sign*(spot*np.exp(-yield_*t)-strike*np.exp(-rate*t)) for t in times))
            if mark < deterministic-tolerance or mark > (spot if sign == 1 else strike)+tolerance:
                raise OptionPricingError("American mark violates intrinsic/upper bounds")
            if not supplied and abs(mark-intrinsic) <= tolerance:
                raise OptionPricingError("American IV is unidentifiable on an exercise plateau; supply an explicit volatility")
        else:
            time = (contract.expirationDate-self.valuationDate).days/365.
            ds, dk = spot*np.exp(-contract.dividendYield*time), strike*np.exp(-self.riskFreeRate*time)
            if mark < max(sign*(ds-dk), 0.)-tolerance or mark > (ds if sign == 1 else dk)+tolerance:
                raise OptionPricingError("European mark violates discounted payoff/upper bounds")
        if supplied:
            return float(self.suppliedVolatilities[index])

        def error(volatility):
            return float(self._context(index, volatility, self.valuationDate).evaluate(spot))-mark

        low, high = self.volatilityBounds
        f_low, f_high = error(low), error(high)
        if abs(f_low) <= tolerance:
            root = low
        elif abs(f_high) <= tolerance:
            root = high
        elif f_low*f_high >= 0:
            raise OptionPricingError("mark has no sign-changing IV bracket in volatilityBounds")
        else:
            root = brentq(error, low, high, xtol=1e-12, rtol=1e-12, maxiter=150)
        if abs(error(root)) > tolerance:
            raise OptionPricingError("excessive final IV calibration residual")
        return float(root)

    def _price(self, index, logReturn, *, derivatives=False):
        with np.errstate(over="raise", invalid="raise"):
            spot = self.spotPrices[self._underlyingIndices[index]]*np.exp(logReturn)
        try:
            return self.contexts[index].evaluate(spot, derivatives=derivatives)
        except ValueError as error:
            # Preserve the nonsmooth exception type for boundary-aware search.
            raise type(error)(f"positions[{index}]: {error}") from error

    def pnl(self, logReturns):
        """Full selected-formula mark change; Ju-Zhong is an American approximation."""
        return super().pnl(logReturns)

    def pnlByStyle(self, logReturns) -> dict[str, np.ndarray]:
        values = self._logReturns(logReturns)
        totals = {style: np.zeros(values.shape[:-1]) for style in ("E", "A")}
        for index, position in enumerate(self.positions):
            quantity = float(position.quantity*position.contract.multiplier)
            if quantity:
                totals[position.contract.exerciseStyle] += quantity*(
                    self._price(index, values[..., self._underlyingIndices[index]])-self.marketPrices[index])
        return totals
