"""Single-dispatch portfolio risk-state BQM encoding and decoding."""

from __future__ import annotations

import math
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any

import numpy

from cache import Cache, CacheFactory

from .risk_state import (
    CorrelatedReturnsVolaGridRiskState,
    CorrelationFactors,
    ReturnsVolaGridRiskState,
    RiskState,
)

if TYPE_CHECKING:
    from portfolio import Portfolio

    from margin_calculator.optimization.optimization_result import BQMOptimizationResult
    from margin_calculator.optimization.optimization_problem.qubo_problem import (
        QUBOProblem,
    )

@dataclass(frozen=True)
class _StructuralQUBOTemplate:
    offsets: numpy.ndarray
    linear: numpy.ndarray
    heads: numpy.ndarray
    tails: numpy.ndarray
    biases: numpy.ndarray
    offset: float


class StructuralQUBOTemplateCache:
    """Typed bounded cache for reusable portfolio-independent QUBO topology."""

    def __init__(
        self,
        memorySize: int = 16,
        cache: Cache[
            tuple[tuple[int, ...], float], _StructuralQUBOTemplate
        ]
        | None = None,
    ) -> None:
        self.cache = cache or CacheFactory.createCache("lru", memorySize)

    def get(
        self,
        key: tuple[tuple[int, ...], float],
    ) -> _StructuralQUBOTemplate | None:
        return self.cache.get(key)

    def insert(
        self,
        key: tuple[tuple[int, ...], float],
        template: _StructuralQUBOTemplate,
    ) -> None:
        self.cache.insert(key, template)


class PortfolioRiskStateBQMVisitor:
    """Encode and decode BQMs using behavior selected by risk-state type."""

    def __init__(
        self,
        structuralCache: StructuralQUBOTemplateCache | None = None,
    ) -> None:
        self.structuralCache = structuralCache or StructuralQUBOTemplateCache()

    @singledispatchmethod
    def createBQM(
        self,
        riskState: RiskState,
        portfolio: Portfolio,
        parameters: Mapping[str, Any],
    ) -> QUBOProblem:
        raise TypeError(f"Unsupported risk state: {type(riskState).__name__}")

    @createBQM.register(ReturnsVolaGridRiskState)
    def _(
        self,
        riskState: ReturnsVolaGridRiskState,
        portfolio: Portfolio,
        parameters: Mapping[str, Any],
    ) -> QUBOProblem:
        return self._createBQM(
            riskState,
            portfolio,
            CorrelationFactors.empty(),
            lambdaOneHot=parameters.get("lambdaOneHot", 1.0),
            lambdaCompat=0.0,
        )

    @createBQM.register(CorrelatedReturnsVolaGridRiskState)
    def _(
        self,
        riskState: CorrelatedReturnsVolaGridRiskState,
        portfolio: Portfolio,
        parameters: Mapping[str, Any],
    ) -> QUBOProblem:
        return self._createBQM(
            riskState,
            portfolio,
            riskState.correlations,
            lambdaOneHot=parameters.get("lambdaOneHot", 1.0),
            lambdaCompat=parameters.get("lambdaCompat", 0.1),
        )

    @singledispatchmethod
    def decodeMargin(
        self,
        riskState: RiskState,
        portfolio: Portfolio,
        bqmOptimizationResult: BQMOptimizationResult,
    ) -> float:
        raise TypeError(f"Unsupported risk state: {type(riskState).__name__}")

    @decodeMargin.register(ReturnsVolaGridRiskState)
    def _(
        self,
        riskState: ReturnsVolaGridRiskState,
        portfolio: Portfolio,
        bqmOptimizationResult: BQMOptimizationResult,
    ) -> float:
        return -self._decodePortfolioReturn(
            riskState,
            portfolio,
            bqmOptimizationResult.sample,
        )

    @staticmethod
    def _decodePortfolioReturn(
        riskState: ReturnsVolaGridRiskState,
        portfolio: Portfolio,
        sample: Mapping[Any, int] | Sequence[int],
    ) -> float:
        """Sum returns for the one selected state of every instrument."""
        if not isinstance(sample, Mapping) and isinstance(sample, (str, bytes)):
            raise TypeError("BQM sample must be a mapping or a binary sequence")

        dense_grid = riskState.returnsVolaGrid
        portfolio_return = 0.0
        variable_position = 0
        for asset, instrument in enumerate(dense_grid.instruments):
            state_returns = dense_grid.gridValues[
                asset,
                dense_grid.validStateMask[asset],
                0,
            ]
            selected_states = []
            for state in range(len(state_returns)):
                variable = f"x_{asset}_{state}"
                value = (
                    sample.get(variable, 0)
                    if isinstance(sample, Mapping)
                    else sample[variable_position]
                )
                if value not in (0, 1, False, True):
                    raise ValueError("BQM sample values must be binary")
                if value:
                    selected_states.append(state)
                variable_position += 1
            if len(selected_states) == 1:
                selected_state = selected_states[0]
            else:
                candidate_states = selected_states or list(
                    range(len(state_returns))
                )
                weight = float(portfolio.weights.get(instrument, 0))
                selected_state = min(
                    candidate_states,
                    key=lambda state: weight * float(state_returns[state]),
                )
            portfolio_return += float(
                portfolio.weights.get(instrument, 0)
            ) * float(state_returns[selected_state])
        return portfolio_return

    def _createBQM(
        self,
        riskState: ReturnsVolaGridRiskState,
        portfolio: Portfolio,
        correlations: CorrelationFactors,
        lambdaOneHot: float,
        lambdaCompat: float,
    ) -> QUBOProblem:
        """Build a QUBO while reusing its state-topology template."""
        for name, value in (
            ("lambdaOneHot", lambdaOneHot),
            ("lambdaCompat", lambdaCompat),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        lambda_one_hot = float(lambdaOneHot)
        lambda_compat = float(lambdaCompat)

        dense_grid = riskState.returnsVolaGrid
        instruments = dense_grid.instruments
        state_counts = tuple(int(count) for count in dense_grid.stateCounts)
        for instrument, state_count in zip(instruments, state_counts):
            if state_count == 0:
                raise ValueError(f"{instrument} has no risk states")

        template_key = (state_counts, lambda_one_hot)
        template = self.structuralCache.get(template_key)
        if template is None:
            offsets = numpy.empty(len(state_counts) + 1, dtype=numpy.int64)
            offsets[0] = 0
            numpy.cumsum(state_counts, out=offsets[1:])
            variable_count = int(offsets[-1])
            one_hot_heads = []
            one_hot_tails = []
            triangle_cache = {}
            for asset, count in enumerate(state_counts):
                triangle = triangle_cache.setdefault(
                    count,
                    numpy.triu_indices(count, k=1),
                )
                one_hot_heads.append(offsets[asset] + triangle[0])
                one_hot_tails.append(offsets[asset] + triangle[1])
            heads = numpy.concatenate(one_hot_heads)
            tails = numpy.concatenate(one_hot_tails)
            template = _StructuralQUBOTemplate(
                offsets=offsets,
                linear=numpy.full(
                    variable_count,
                    -lambda_one_hot,
                    dtype=float,
                ),
                heads=heads,
                tails=tails,
                biases=numpy.full(
                    len(heads),
                    2.0 * lambda_one_hot,
                    dtype=float,
                ),
                offset=lambda_one_hot * len(state_counts),
            )
            self.structuralCache.insert(template_key, template)

        offsets = template.offsets
        portfolio_weights = numpy.fromiter(
            (
                float(portfolio.weights.get(instrument, 0))
                for instrument in instruments
            ),
            dtype=float,
            count=len(instruments),
        )
        if not numpy.isfinite(portfolio_weights).all():
            raise ValueError("portfolio contains a non-finite weight")
        weighted_returns = (
            portfolio_weights[:, None] * dense_grid.gridValues[:, :, 0]
        )
        linear = template.linear + weighted_returns[
            dense_grid.validStateMask
        ]

        if len(correlations) and lambda_compat != 0.0:
            asset_count = len(state_counts)
            if (
                numpy.any(correlations.firstAssets >= asset_count)
                or numpy.any(correlations.secondAssets >= asset_count)
                or numpy.any(correlations.firstAssets < 0)
                or numpy.any(correlations.secondAssets < 0)
            ):
                raise ValueError("correlation factors contain an unknown asset")
            first_limits = numpy.asarray(state_counts)[correlations.firstAssets]
            second_limits = numpy.asarray(state_counts)[correlations.secondAssets]
            if (
                numpy.any(correlations.firstStates >= first_limits)
                or numpy.any(correlations.secondStates >= second_limits)
                or numpy.any(correlations.firstStates < 0)
                or numpy.any(correlations.secondStates < 0)
            ):
                raise ValueError("correlation factors contain an unknown state")
            correlation_heads = (
                offsets[correlations.firstAssets] + correlations.firstStates
            )
            correlation_tails = (
                offsets[correlations.secondAssets] + correlations.secondStates
            )
            heads = numpy.concatenate((template.heads, correlation_heads))
            tails = numpy.concatenate((template.tails, correlation_tails))
            biases = numpy.concatenate(
                (
                    template.biases,
                    lambda_compat * correlations.coefficients,
                )
            )
        else:
            heads = template.heads
            tails = template.tails
            biases = template.biases

        from margin_calculator.optimization.optimization_problem.qubo_problem import (
            QUBOProblem,
        )

        return QUBOProblem(
            linear=linear,
            quadraticHeads=heads,
            quadraticTails=tails,
            quadraticBiases=biases,
            offset=template.offset,
            groupOffsets=template.offsets,
            seedOffset=self._stableSeedOffset(
                instruments,
                linear,
                correlations,
                lambda_one_hot,
                lambda_compat,
            ),
        )

    @staticmethod
    def _stableSeedOffset(
        instruments: Sequence[Any],
        linear: numpy.ndarray,
        correlations: CorrelationFactors,
        lambdaOneHot: float,
        lambdaCompat: float,
    ) -> int:
        """Hash scenario-specific values without re-hashing cached topology."""
        digest = hashlib.blake2b(digest_size=8, person=b"QUBOseed")
        for instrument in instruments:
            encoded = str(instrument).encode("utf-8")
            digest.update(len(encoded).to_bytes(4, "little"))
            digest.update(encoded)
        digest.update(memoryview(numpy.ascontiguousarray(linear)).cast("B"))
        for values in (
            correlations.firstAssets,
            correlations.firstStates,
            correlations.secondAssets,
            correlations.secondStates,
            correlations.coefficients,
        ):
            digest.update(memoryview(numpy.ascontiguousarray(values)).cast("B"))
        digest.update(
            numpy.asarray(
                [lambdaOneHot, lambdaCompat],
                dtype=numpy.float64,
            ).tobytes()
        )
        return int.from_bytes(digest.digest(), "little")
