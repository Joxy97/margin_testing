"""European equity options calibrated to marks and repriced at fixed implied IV."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np

from portfolio import DerivativePosition, EquityOptionContract
from .models import EquityBlackScholesPricingModel, impliedVolatility


def _owned(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("European option arrays must be finite")
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


@dataclass(frozen=True)
class EuropeanOptionBook:
    """One dated market snapshot; marks are quote prices, quantities are contracts.

    Spot prices follow instruments; market prices follow positions. All marks
    must be observed on valuationDate. Both calibration and horizon repricing
    use ACT/365. IV is calibrated per contract, then held fixed at its strike.
    """

    instruments: tuple[str, ...]
    spotPrices: np.ndarray
    positions: tuple[DerivativePosition, ...]
    marketPrices: np.ndarray
    valuationDate: date
    horizonDate: date
    riskFreeRate: float = 0.0
    impliedVolatilities: np.ndarray = field(init=False)
    _underlyingIndices: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "instruments", tuple(self.instruments))
        object.__setattr__(self, "positions", tuple(self.positions))
        for name in ("spotPrices", "marketPrices"):
            object.__setattr__(self, name, _owned(getattr(self, name)))
        if (not self.instruments or len(set(self.instruments)) != len(self.instruments)
                or any(not isinstance(i, str) or not i for i in self.instruments)):
            raise ValueError("option instruments must be nonempty, unique symbol strings")
        if self.spotPrices.shape != (len(self.instruments),) or np.any(self.spotPrices <= 0):
            raise ValueError("spotPrices must be positive and match instruments")
        if self.marketPrices.shape != (len(self.positions),) or np.any(self.marketPrices < 0):
            raise ValueError("marketPrices must be nonnegative and match positions")
        if type(self.valuationDate) is not date or type(self.horizonDate) is not date:
            raise TypeError("valuationDate and horizonDate must be dates")
        if self.horizonDate < self.valuationDate:
            raise ValueError("horizonDate must not precede valuationDate")
        if isinstance(self.riskFreeRate, bool) or not isinstance(self.riskFreeRate, (int, float)):
            raise TypeError("riskFreeRate must be numeric")
        if not np.isfinite(self.riskFreeRate):
            raise ValueError("riskFreeRate must be finite")
        indices, volatilities = [], []
        for index, position in enumerate(self.positions):
            if not isinstance(position, DerivativePosition):
                raise TypeError(f"positions[{index}] must be a DerivativePosition")
            contract = position.contract
            if not isinstance(contract, EquityOptionContract):
                raise TypeError(f"positions[{index}] must contain an equity option")
            self._validateContract(index, contract)
            if contract.symbol not in self.instruments:
                raise ValueError(f"positions[{index}] underlying is absent from instruments")
            if type(contract.expirationDate) is not date:
                raise TypeError(f"positions[{index}] expirationDate must be a date")
            if contract.expirationDate <= self.valuationDate:
                raise ValueError(f"positions[{index}] must be unexpired on valuationDate")
            if contract.expirationDate < self.horizonDate:
                raise ValueError(f"positions[{index}] expires before horizonDate; settlement needs a price path")
            if not np.isfinite(float(position.quantity * contract.multiplier)):
                raise ValueError(f"positions[{index}] scaled quantity exceeds float64")
            if not np.isfinite(float(contract.strike)):
                raise ValueError(f"positions[{index}] strike exceeds float64")
            underlying = self.instruments.index(contract.symbol)
            indices.append(underlying)
            try:
                volatility = self._calibrate(index, underlying)
            except ValueError as error:
                raise ValueError(f"positions[{index}] implied volatility: {error}") from error
            volatilities.append(volatility)
        object.__setattr__(self, "_underlyingIndices", tuple(indices))
        object.__setattr__(self, "impliedVolatilities", _owned(volatilities))

    def _validateContract(self, index: int, contract: EquityOptionContract) -> None:
        if contract.exerciseStyle != "E":
            raise ValueError(f"positions[{index}] must be European; American options are unsupported")

    def _calibrate(self, index: int, underlying: int) -> float:
        contract = self.positions[index].contract
        return impliedVolatility(
            EquityBlackScholesPricingModel(), float(self.marketPrices[index]),
            float(self.spotPrices[underlying]), float(contract.strike),
            (contract.expirationDate-self.valuationDate).days/365., self.riskFreeRate,
            contract.optionType, contract.dividendYield, maximumIterations=100,
        )

    def _logReturns(self, logReturns) -> np.ndarray:
        values = np.asarray(logReturns, dtype=np.float64)
        if values.ndim < 1 or values.shape[-1] != len(self.instruments) or not np.isfinite(values).all():
            raise ValueError("logReturns must be finite with instruments on the last axis")
        return values

    def _price(self, index: int, logReturn: np.ndarray, *, derivatives: bool = False):
        """Vectorized BS price and, when requested, derivatives in log spot."""
        from scipy.special import ndtr

        contract = self.positions[index].contract
        underlying = self._underlyingIndices[index]
        time = (contract.expirationDate-self.horizonDate).days/365.
        strike = float(contract.strike)
        sign = 1. if contract.optionType == "C" else -1.
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            spot = self.spotPrices[underlying] * np.exp(logReturn)
            if time == 0:
                payoff = np.maximum(sign*(spot-strike), 0.)
                if not derivatives:
                    return payoff
                if np.any(spot == strike):
                    raise ValueError("expiry payoff is not differentiable at the strike; quadratic expansion is undefined")
                first = np.where(sign*(spot-strike) > 0, sign*spot, 0.)
                return payoff, first, first
            sigma = self.impliedVolatilities[index]
            root = sigma*np.sqrt(time)
            d1 = (np.log(self.spotPrices[underlying]/strike) + logReturn
                  + (self.riskFreeRate-contract.dividendYield+.5*sigma*sigma)*time)/root
            d2 = d1-root
            discounted_spot = spot*np.exp(-contract.dividendYield*time)
            discounted_strike = strike*np.exp(-self.riskFreeRate*time)
            price = sign*(discounted_spot*ndtr(sign*d1)-discounted_strike*ndtr(sign*d2))
            if not derivatives:
                return price
            first = sign*discounted_spot*ndtr(sign*d1)
            second = first + discounted_spot*np.exp(-.5*d1*d1)/np.sqrt(2*np.pi)/root
            return price, first, second

    def pnl(self, logReturns: np.ndarray) -> np.ndarray:
        """Exact BS horizon value minus the observed mark, including multipliers."""
        values = self._logReturns(logReturns)
        result = np.zeros(values.shape[:-1], dtype=np.float64)
        for index, (position, underlying) in enumerate(zip(self.positions, self._underlyingIndices)):
            quantity = float(position.quantity*position.contract.multiplier)
            if quantity:
                result += quantity*(self._price(index, values[..., underlying])-self.marketPrices[index])
        if not np.isfinite(result).all():
            raise FloatingPointError("European option P&L exceeds float64")
        return result

    def logDerivatives(self, logReturns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Aggregate S*delta and S*delta + S**2*gamma by underlying."""
        values = self._logReturns(logReturns)
        if values.ndim != 1:
            raise ValueError("logDerivatives requires one log-return vector")
        first, second = np.zeros(len(self.instruments)), np.zeros(len(self.instruments))
        for index, (position, underlying) in enumerate(zip(self.positions, self._underlyingIndices)):
            quantity = float(position.quantity*position.contract.multiplier)
            if quantity:
                _, dx, dxx = self._price(index, values[underlying], derivatives=True)
                first[underlying] += quantity*dx
                second[underlying] += quantity*dxx
        if not (np.isfinite(first).all() and np.isfinite(second).all()):
            raise FloatingPointError("European option derivatives exceed float64")
        return first, second
