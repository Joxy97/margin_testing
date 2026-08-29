"""Portfolio risk-state BQM encoding and decoding."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy

from cache import Cache, CacheFactory

from .risk_state import CorrelationFactors

if TYPE_CHECKING:
    from portfolio import Portfolio

    from margin_calculator.optimization.optimization_result import BQMOptimizationResult
    from margin_calculator.optimization.optimization_problem.qubo_problem import (
        QUBOProblem,
    )

    from .risk_state import (
        CorrelatedReturnsVolaGridRiskState,
        ReturnsVolaGridRiskState,
    )


@dataclass(frozen=True)
class _StructuralQUBOTemplate:
    offsets: numpy.ndarray
    linear: numpy.ndarray
    heads: numpy.ndarray
    tails: numpy.ndarray
    biases: numpy.ndarray
    offset: float
    oneHotGroups: tuple[tuple[int, ...], ...]


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


class PortfolioRiskStateBQMManager:
    """Encode portfolio risk states as BQMs and decode solver results."""

    def __init__(
        self,
        structuralCache: StructuralQUBOTemplateCache | None = None,
    ) -> None:
        self.structuralCache = structuralCache or StructuralQUBOTemplateCache()

    def createReturnsVolaGridBQM(
        self,
        riskState: ReturnsVolaGridRiskState,
        portfolio: Portfolio,
        parameters: Mapping[str, Any],
    ) -> QUBOProblem:
        """Create a BQM for a returns-volatility-grid risk state."""
        return self._createBQM(
            riskState,
            portfolio,
            CorrelationFactors.empty(),
            lambdaOneHot=parameters.get("lambdaOneHot", 1.0),
            lambdaCompat=0.0,
        )

    def createCorrelatedReturnsVolaGridBQM(
        self,
        riskState: CorrelatedReturnsVolaGridRiskState,
        portfolio: Portfolio,
        parameters: Mapping[str, Any],
    ) -> QUBOProblem:
        """Create a BQM for a correlated returns-volatility-grid risk state."""
        return self._createBQM(
            riskState,
            portfolio,
            riskState.correlations,
            lambdaOneHot=parameters.get("lambdaOneHot", 1.0),
            lambdaCompat=parameters.get("lambdaCompat", 0.1),
        )

    def decodeReturnsVolaGridRiskState(
        self,
        riskState: ReturnsVolaGridRiskState,
        portfolio: Portfolio,
        bqmOptimizationResult: BQMOptimizationResult,
    ) -> float:
        """Decode the selected states into a positive portfolio loss."""
        return -self._decodePortfolioReturn(
            riskState,
            portfolio,
            bqmOptimizationResult.sample,
        )

    def decodeCorrelatedReturnsVolaGridRiskState(
        self,
        riskState: CorrelatedReturnsVolaGridRiskState,
        portfolio: Portfolio,
        bqmOptimizationResult: BQMOptimizationResult,
    ) -> float:
        """Decode a correlated BQM result into a positive portfolio loss."""
        return self.decodeReturnsVolaGridRiskState(
            riskState,
            portfolio,
            bqmOptimizationResult,
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

        portfolio_return = 0.0
        variable_position = 0
        for asset, (instrument, state_grid) in enumerate(
            riskState.returnsVolaGrid.items()
        ):
            selected_states = []
            for state in range(len(state_grid)):
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
                candidate_states = selected_states or list(range(len(state_grid)))
                weight = float(portfolio.weights.get(instrument, 0))
                selected_state = min(
                    candidate_states,
                    key=lambda state: weight * float(state_grid[state, 0]),
                )
            portfolio_return += float(
                portfolio.weights.get(instrument, 0)
            ) * float(state_grid[selected_state, 0])
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

        instruments = tuple(riskState.returnsVolaGrid)
        state_counts = tuple(
            len(riskState.returnsVolaGrid[instrument])
            for instrument in instruments
        )
        for instrument, state_count in zip(instruments, state_counts):
            grid = riskState.returnsVolaGrid[instrument]
            if grid.ndim != 2 or grid.shape[1] < 1:
                raise ValueError(
                    f"{instrument} risk states must be a two-dimensional grid"
                )
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
                oneHotGroups=tuple(
                    tuple(range(int(offsets[asset]), int(offsets[asset + 1])))
                    for asset in range(len(state_counts))
                ),
            )
            self.structuralCache.insert(template_key, template)

        offsets = template.offsets
        portfolio_linear_parts = []
        for instrument in instruments:
            weight = float(portfolio.weights.get(instrument, 0))
            grid_returns = numpy.asarray(
                riskState.returnsVolaGrid[instrument][:, 0],
                dtype=float,
            )
            if not math.isfinite(weight):
                raise ValueError(f"{instrument} has a non-finite portfolio weight")
            if not numpy.isfinite(grid_returns).all():
                raise ValueError(f"{instrument} contains a non-finite return")
            portfolio_linear_parts.append(weight * grid_returns)
        linear = template.linear + numpy.concatenate(portfolio_linear_parts)

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
            oneHotGroups=template.oneHotGroups,
        )
