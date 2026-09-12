"""Fixed-parameter vanilla valuation; Ju–Zhong equations from the supplied audit.

American values approximate continuous exercise with continuous dividend yield.
Boundary contexts are independent of scenario spot. No clipping or spot-specific
backend switching is performed. See docs/benchmarks/american_factor_stress.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

from .models import OptionPricingModel


class OptionPricingError(ValueError):
    """The selected formula cannot price the requested domain reliably."""


class NonsmoothOptionError(OptionPricingError):
    """Two-sided log-spot derivatives are undefined at this price."""


@dataclass(frozen=True)
class VanillaPriceContext:
    """Immutable positive-IV context, reusable for scalar or batched spots."""

    strike: float
    timeToExpiry: float
    riskFreeRate: float
    dividendYield: float
    volatility: float
    optionType: str
    exerciseStyle: str = "E"
    boundary: float | None = field(init=False, default=None)
    premium: float = field(init=False, default=0.)
    eta: float = field(init=False, default=0.)
    b: float = field(init=False, default=0.)
    c: float = field(init=False, default=0.)
    boundaryResidual: float = field(init=False, default=0.)

    def __post_init__(self) -> None:
        for name in ("strike", "timeToExpiry", "riskFreeRate", "dividendYield", "volatility"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.strike <= 0 or self.volatility <= 0 or self.timeToExpiry < 0:
            raise ValueError("strike/volatility must be positive; maturity must be nonnegative")
        if self.optionType not in ("C", "P") or self.exerciseStyle not in ("E", "A"):
            raise ValueError("optionType must be C/P and exerciseStyle E/A")
        if self.exerciseStyle == "A" and (self.riskFreeRate < 0 or self.dividendYield < 0):
            raise OptionPricingError("Ju-Zhong requires nonnegative rate and continuous dividend yield")
        if self.timeToExpiry == 0 or self.exerciseStyle == "E" or self.europeanEquivalent:
            return
        t, r, q, v = self.timeToExpiry, self.riskFreeRate, self.dividendYield, self.volatility
        beta = 2*(r-q)/(v*v)
        kappa = 2/(v*v*t) if r == 0 else 2*r/(v*v*(-np.expm1(-r*t)))
        zeta = np.hypot(beta-1, 2*np.sqrt(kappa))
        # Compute the large root first; obtain the other through their product.
        large = -.5*((beta-1)+np.copysign(zeta, beta-1))
        small = -kappa/large
        eta = max(large, small) if self.sign == 1 else min(large, small)

        def residual(y):
            spot = self.strike*np.exp(y)
            value, first, _, _ = self._european(spot)
            premium = self.sign*(spot-self.strike)-value
            return float(first/spot+eta*premium/spot-self.sign)

        endpoint, f0 = 0., residual(0.)
        for magnitude in np.geomspace(.01, 30., 100):
            endpoint = self.sign*float(magnitude)
            if residual(endpoint)*f0 < 0:
                break
        else:
            raise OptionPricingError("Ju-Zhong boundary has no finite sign-changing bracket")
        root = brentq(residual, min(0., endpoint), max(0., endpoint), xtol=1e-13, rtol=1e-14)
        boundary = self.strike*np.exp(root)
        value, _, _, maturity_derivative = self._european(boundary)
        premium = float(self.sign*(boundary-self.strike)-value)
        error = residual(root)
        if not np.isfinite(premium) or premium <= 1e-14*self.strike or abs(error) > 1e-8:
            raise OptionPricingError("Ju-Zhong boundary premium or root residual is invalid")
        b = -np.exp(-r*t)*kappa*kappa/(2*zeta*zeta)
        c = -(2*maturity_derivative/(v*v*premium)+np.exp(-r*t)*kappa+2*b)/(self.sign*zeta)
        for name, value in (("boundary", boundary), ("premium", premium), ("eta", eta),
                            ("b", b), ("c", c), ("boundaryResidual", error)):
            if not np.isfinite(value):
                raise OptionPricingError(f"nonfinite Ju-Zhong {name}")
            object.__setattr__(self, name, float(value))

    @property
    def sign(self) -> float:
        return 1. if self.optionType == "C" else -1.

    @property
    def europeanEquivalent(self) -> bool:
        return ((self.optionType == "C" and self.dividendYield == 0)
                or (self.optionType == "P" and self.riskFreeRate == 0))

    @property
    def backend(self) -> str:
        if self.timeToExpiry == 0:
            return "payoff"
        return "ju_zhong" if self.boundary is not None else "black_scholes"

    @property
    def kink(self) -> float | None:
        return self.strike if self.timeToExpiry == 0 else self.boundary

    def _european(self, spot):
        t, r, q, v = self.timeToExpiry, self.riskFreeRate, self.dividendYield, self.volatility
        root = v*np.sqrt(t)
        d1 = (np.log(spot/self.strike)+(r-q+.5*v*v)*t)/root
        d2 = d1-root
        ds, dk = spot*np.exp(-q*t), self.strike*np.exp(-r*t)
        density = np.exp(-.5*d1*d1)/np.sqrt(2*np.pi)
        value = self.sign*(ds*ndtr(self.sign*d1)-dk*ndtr(self.sign*d2))
        first = self.sign*ds*ndtr(self.sign*d1)
        second = first+ds*density/root
        maturity = ds*density*v/(2*np.sqrt(t))-q*first+self.sign*r*dk*ndtr(self.sign*d2)
        return value, first, second, maturity

    def evaluate(self, spots, *, derivatives: bool = False):
        """Return value, or value and its two log-spot derivatives on smooth branches."""
        spot = np.asarray(spots, dtype=np.float64)
        if not np.isfinite(spot).all() or np.any(spot <= 0):
            raise ValueError("spots must be positive and finite")
        if derivatives and self.kink is not None and np.any(np.abs(np.log(spot/self.kink)) <= 1e-10):
            raise NonsmoothOptionError("option is not differentiable at exercise boundary/expiry strike")
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            if self.timeToExpiry == 0:
                value = np.maximum(self.sign*(spot-self.strike), 0.)
                first = np.where(value > 0, self.sign*spot, 0.)
                return (value, first, first) if derivatives else value
            european, first, second, _ = self._european(spot)
            value = np.array(european, copy=True)
            if self.boundary is not None:
                continuation = self.sign*(self.boundary-spot) > 0
                # Evaluate correction only on continuation spots, avoiding invalid
                # powers/denominators on the unused exercise branch.
                x = np.log(spot[continuation]/self.boundary)
                denominator = 1-self.b*x*x-self.c*x
                if np.any(denominator <= 1e-10):
                    raise OptionPricingError("Ju-Zhong continuation denominator is nonpositive or unsafe")
                premium = self.premium*np.exp(self.eta*x)/denominator
                u = self.c+2*self.b*x
                slope = self.eta+u/denominator
                value[continuation] += premium
                first = np.array(first, copy=True)
                second = np.array(second, copy=True)
                first[continuation] += premium*slope
                second[continuation] += premium*(slope*slope+2*self.b/denominator+(u/denominator)**2)
                value[~continuation] = self.sign*(spot[~continuation]-self.strike)
                first[~continuation] = self.sign*spot[~continuation]
                second[~continuation] = self.sign*spot[~continuation]
            if self.exerciseStyle == "A":
                lower = np.maximum(np.maximum(self.sign*(spot-self.strike), 0.), european)
                upper = spot if self.sign == 1 else self.strike
                tolerance = 1e-7*np.maximum(spot, self.strike)
                if np.any(value < lower-tolerance) or np.any(value > upper+tolerance):
                    raise OptionPricingError("Ju-Zhong price materially violates intrinsic/European/upper bounds")
            if not all(np.isfinite(a).all() for a in (value, first, second)):
                raise OptionPricingError("nonfinite option price or derivatives")
            return (value, first, second) if derivatives else value

    def screenInterval(self, low: float, high: float) -> dict:
        """Exact denominator minimum; sampled economic checks are not a proof."""
        if not (np.isfinite(low) and np.isfinite(high) and 0 < low <= high):
            raise ValueError("screen interval must have finite positive ordered endpoints")
        minimum = None
        if self.boundary is not None:
            left, right = np.log(low/self.boundary), np.log(high/self.boundary)
            left, right = (left, min(right, 0.)) if self.sign == 1 else (max(left, 0.), right)
            if left <= right:
                candidates = [left, right]
                vertex = -self.c/(2*self.b)
                if left <= vertex <= right:
                    candidates.append(vertex)
                minimum = float(min(1-self.b*x*x-self.c*x for x in candidates))
                if minimum <= 1e-10:
                    raise OptionPricingError("Ju-Zhong interval contains an unsafe continuation denominator")
        self.evaluate(np.geomspace(low, high, 65))
        return {"spot_low": low, "spot_high": high, "denominator_minimum": minimum,
                "economic_check_points": 65, "boundary": self.kink, "backend": self.backend}


class JuZhongPricingModel(OptionPricingModel):
    """Scalar adapter; scenario code should reuse VanillaPriceContext instead."""

    def price(self, underlyingPrice, strike, timeToExpiry, riskFreeRate,
              volatility, optionType, dividendYield=0.) -> float:
        return float(VanillaPriceContext(strike, timeToExpiry, riskFreeRate,
            dividendYield, volatility, optionType, "A").evaluate(underlyingPrice))
