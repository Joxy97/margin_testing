"""Past-only rolling volatility forecasts and fixed-geometry stress overlays."""
from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import date
from collections.abc import Mapping

import numpy as np

from .factor_stress_model import FactorStressModel, _owned


@dataclass(frozen=True)
class VolatilityConfig:
    """Universe-pooled preset dynamics; calendar coefficients fit prior data only."""
    name: str
    kind: str = 'baseline'
    decay: float = .93
    slowDecay: float = .97
    alpha: float = .04
    beta: float = .90
    gamma: float = .08
    calendar: str = 'none'
    calendarRidge: float = 10.
    jumpThreshold: float = 2.5
    jumpHalfLife: float = 5.
    jumpGain: float = 0.
    conservative: bool = True

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError('volatility name must be nonempty')
        if self.kind not in {'baseline', 'ewma', 'envelope', 'garch', 'gjr'}:
            raise ValueError('unsupported volatility kind')
        if self.calendar not in {'none', 'weekday', 'weekday_gap', 'fourier'}:
            raise ValueError('unsupported calendar model')
        if not isinstance(self.conservative, bool):
            raise TypeError('conservative must be boolean')
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name not in {'name', 'kind', 'calendar', 'conservative'}:
                if isinstance(value, bool) or not isinstance(value, (float, int)):
                    raise TypeError(f'{field.name} must be numeric')
                if not np.isfinite(value):
                    raise ValueError(f'{field.name} must be finite')
        if not 0 < self.decay < 1 or not 0 < self.slowDecay < 1:
            raise ValueError('EWMA decays must lie strictly between zero and one')
        persistence = self.alpha+self.beta+(self.gamma/2 if self.kind == 'gjr' else 0)
        if min(self.alpha, self.beta, self.gamma) < 0 or persistence >= 1:
            raise ValueError('GARCH coefficients must be nonnegative and stationary')
        if min(self.calendarRidge, self.jumpThreshold, self.jumpHalfLife) <= 0 or self.jumpGain < 0:
            raise ValueError('ridge, threshold, half-life must be positive; jump gain nonnegative')
        if (self.calendar != 'none' or self.jumpGain) and self.kind != 'gjr':
            raise ValueError('calendar and jump overlays require GJR')

    @classmethod
    def fromMapping(cls, value: Mapping) -> VolatilityConfig:
        if not isinstance(value, Mapping):
            raise TypeError('volatility config must be a mapping')
        unknown = set(value)-{f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f'unknown volatility settings: {sorted(unknown)}')
        return cls(**value)


@dataclass(frozen=True)
class VolatilityForecast:
    variance: np.ndarray
    ratio: np.ndarray
    jumpState: np.ndarray
    calendarMultiplier: float
    calendarCoefficients: np.ndarray

    def __post_init__(self):
        for name in ('variance', 'ratio', 'jumpState', 'calendarCoefficients'):
            object.__setattr__(self, name, _owned(getattr(self, name)))


def variancePath(innovations, target, *, alpha, beta, gamma=0.):
    """Return h before each innovation and the final one-step forecast.

    A rolling-window backcast initializes h to the forecast-origin target.
    The entire input window must precede the forecast date. This is not a
    sequence of historical out-of-sample forecasts within that window.
    """
    eps, target = np.asarray(innovations, dtype=float), np.asarray(target, dtype=float)
    if eps.ndim != 2 or target.shape != (eps.shape[1],) or not len(eps):
        raise ValueError('innovations and target must have matching asset shapes')
    if not np.isfinite(eps).all() or not np.isfinite(target).all() or np.any(target <= 0):
        raise ValueError('innovations must be finite and target strictly positive')
    if not np.isfinite([alpha, beta, gamma]).all() or min(alpha, beta, gamma) < 0 or alpha+beta+gamma/2 > 1+1e-15:
        raise ValueError('invalid variance recursion coefficients')
    h = target.copy()
    path = np.empty_like(eps)
    intercept = max(0., 1-alpha-beta-gamma/2)*target
    for t, innovation in enumerate(eps):
        path[t] = h
        h = intercept + (alpha+gamma*(innovation < 0))*innovation**2 + beta*h
    return path, h


def calendarFeatures(days, gaps, kind):
    weekday = np.array([day.weekday() for day in days])
    if kind == 'fourier':
        return np.column_stack((np.cos(2*np.pi*weekday/5), np.sin(2*np.pi*weekday/5)))
    columns = [(weekday == 0).astype(float), (weekday == 4).astype(float)]
    if kind == 'weekday_gap':
        columns.append(np.asarray(gaps, dtype=float)-1)
    return np.column_stack(columns)


def forecastVolatility(logReturns, closeDates, forecastDate, center, baselineScale,
                       config: VolatilityConfig) -> VolatilityForecast:
    """Forecast from one complete prior-close window, never the evaluation close.

    Dynamics use fixed universe-pooled presets, not in-sample breach tuning.
    Calendar ridge fits log pooled squared standardized innovations; centered
    features normalize the multiplier to geometric mean one on the fit window.
    Calendar gaps are from this market panel, not inferred exchange holidays.
    """
    returns = np.asarray(logReturns, dtype=float)
    center, scale = np.asarray(center, dtype=float), np.asarray(baselineScale, dtype=float)
    days = tuple(date.fromisoformat(str(day)[:10]) for day in closeDates)
    future = date.fromisoformat(str(forecastDate)[:10])
    if returns.ndim != 2 or len(days) != len(returns)+1 or len(returns) < 2:
        raise ValueError('returns need one more prior close date than observations')
    if any(a >= b for a,b in zip(days, days[1:])) or days[-1] >= future:
        raise ValueError('close dates must increase strictly and precede forecast date')
    if center.shape != (returns.shape[1],) or scale.shape != center.shape:
        raise ValueError('center and baseline scale must match asset ordering')
    if not np.isfinite(returns).all() or not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('finite inputs and positive baseline scales required')
    eps, target = returns-center, scale**2
    jump = np.zeros_like(target)
    coefficients = np.empty(0)
    multiplier = 1.
    if config.kind == 'baseline':
        forecast = target.copy()
    elif config.kind in {'ewma', 'envelope'}:
        _, forecast = variancePath(eps, target, alpha=1-config.decay, beta=config.decay)
        if config.kind == 'envelope':
            _, slow = variancePath(eps, target, alpha=1-config.slowDecay, beta=config.slowDecay)
            forecast = np.maximum(forecast, slow)
    else:
        path, forecast = variancePath(eps, target, alpha=config.alpha, beta=config.beta,
            gamma=config.gamma if config.kind == 'gjr' else 0.)
        standardized2 = eps**2/path
        if config.calendar != 'none':
            gaps = [(b-a).days for a,b in zip(days, days[1:])]
            x = calendarFeatures(days[1:], gaps, config.calendar)
            average = x.mean(axis=0)
            centered = x-average
            # Tiny only prevents log(0) for a zero pooled innovation.
            response = np.log(np.maximum(standardized2.mean(axis=1), np.finfo(float).tiny))
            coefficients = np.linalg.solve(centered.T@centered+config.calendarRidge*np.eye(x.shape[1]),
                                            centered.T@(response-response.mean()))
            next_x = calendarFeatures([future], [(future-days[-1]).days], config.calendar)[0]
            multiplier = float(np.exp((next_x-average)@coefficients))
            forecast *= multiplier
        if config.jumpGain:
            decay = 2**(-1/config.jumpHalfLife)
            for z2 in standardized2:
                jump = decay*jump+np.maximum(z2-config.jumpThreshold**2, 0.)
            forecast *= 1+config.jumpGain*jump
    ratio = np.sqrt(forecast)/scale
    if config.conservative:
        ratio = np.maximum(1., ratio)
    # Algebraically identical normalized-EW/backcast variances can differ by
    # rounding. Canonicalize unit overlays before nonlinear optimization.
    ratio[np.abs(ratio-1.) <= 32*np.finfo(float).eps] = 1.
    return VolatilityForecast(forecast, ratio, jump, multiplier, coefficients)


def rescaleStressModel(model: FactorStressModel, ratio) -> FactorStressModel:
    ratio = np.asarray(ratio, dtype=float)
    if ratio.shape != model.center.shape or not np.isfinite(ratio).all() or np.any(ratio <= 0):
        raise ValueError('volatility ratios must be positive, finite and match assets')
    return FactorStressModel(model.instruments, model.exposures, model.center, model.directions*ratio[:, None])
