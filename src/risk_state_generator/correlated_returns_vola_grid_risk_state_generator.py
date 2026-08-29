"""Correlated returns-volatility-grid risk-state generator."""

from numbers import Integral
from typing import Any

import numpy

from .pca_grid import ReturnsPCAGrid
from .returns_vola_grid_risk_state_generator import (
    ReturnsVolaGridRiskStateGenerator,
    _ConditionedRiskState,
)
from .risk_state import CorrelatedReturnsVolaGridRiskState, CorrelationFactors


class CorrelatedReturnsVolaGridRiskStateGenerator(
    ReturnsVolaGridRiskStateGenerator
):
    """Add state-pair residual compatibility to returns-volatility grids."""

    def __init__(
        self,
        *args: Any,
        topKNeighbors: int = 5,
        correlationBlockBytes: int = 128 * 1024 * 1024,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if (
            isinstance(topKNeighbors, bool)
            or not isinstance(topKNeighbors, Integral)
            or topKNeighbors < 0
        ):
            raise ValueError("topKNeighbors must be a nonnegative integer")
        if (
            isinstance(correlationBlockBytes, bool)
            or not isinstance(correlationBlockBytes, Integral)
            or correlationBlockBytes <= 0
        ):
            raise ValueError("correlationBlockBytes must be a positive integer")
        self.topKNeighbors = int(topKNeighbors)
        self.correlationBlockBytes = int(correlationBlockBytes)

    def _createRiskState(
        self,
        pcaGrid: ReturnsPCAGrid,
        conditionedState: _ConditionedRiskState,
    ) -> CorrelatedReturnsVolaGridRiskState:
        """Add conditional correlations to a conditioned risk state."""
        return CorrelatedReturnsVolaGridRiskState(
            returnsVolaGrid=conditionedState.returnsVolaGrid,
            correlations=self._buildCorrelations(
                pcaGrid,
                conditionedState,
            ),
        )

    def _buildCorrelations(
        self,
        pcaGrid: ReturnsPCAGrid,
        conditionedState: _ConditionedRiskState,
    ) -> CorrelationFactors:
        """Build compact MarginLab compatibility coefficients."""
        inflated_residuals = (
            conditionedState.nearestResiduals
            * conditionedState.inflationFactor
        )
        conditional_std, neighbor_indices, neighbor_correlations = (
            self._getConditionalNeighbors(
                inflated_residuals,
                conditionedState.residualSigma,
            )
        )
        standardized_grids = self._getStandardizedGrids(
            pcaGrid,
            conditionedState,
        )
        return self._getStatePairCoefficients(
            standardized_grids,
            conditionedState.scenarioCenter,
            conditional_std,
            neighbor_indices,
            neighbor_correlations,
        )

    def _getConditionalNeighbors(
        self,
        residualSamples: numpy.ndarray,
        fallbackSigma: numpy.ndarray,
    ) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """Find exact top-k correlations using bounded row blocks."""
        observations, assets = residualSamples.shape
        neighbor_count = min(self.topKNeighbors, max(assets - 1, 0))
        neighbor_indices = numpy.empty(
            (assets, neighbor_count),
            dtype=numpy.int32,
        )
        neighbor_correlations = numpy.empty(
            (assets, neighbor_count),
            dtype=numpy.float64,
        )

        if observations > 1:
            centered = residualSamples - residualSamples.mean(axis=0)
            covariance_scale = observations - 1
            variances = numpy.sum(centered * centered, axis=0) / covariance_scale
        else:
            centered = numpy.zeros_like(residualSamples)
            covariance_scale = 1
            variances = numpy.zeros(assets, dtype=float)
        valid_variances = numpy.isfinite(variances) & (variances > 1e-12)
        variances = numpy.where(
            valid_variances,
            variances,
            numpy.maximum(fallbackSigma**2, 1e-12),
        )
        standard_deviation = numpy.sqrt(variances)
        if neighbor_count == 0:
            return standard_deviation, neighbor_indices, neighbor_correlations

        normalized = centered / standard_deviation
        rows_per_block = max(
            1,
            min(assets, self.correlationBlockBytes // (16 * assets)),
        )
        for start in range(0, assets, rows_per_block):
            stop = min(start + rows_per_block, assets)
            correlations = (
                normalized[:, start:stop].T
                @ normalized
                / covariance_scale
            )
            correlations = numpy.clip(
                numpy.nan_to_num(
                    correlations,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ),
                -0.999,
                0.999,
            )
            absolute = numpy.abs(correlations)
            local_rows = numpy.arange(stop - start)
            absolute[local_rows, numpy.arange(start, stop)] = -numpy.inf

            if assets <= 512:
                chosen = numpy.argsort(absolute, axis=1)[:, -neighbor_count:]
            else:
                chosen = numpy.argpartition(
                    absolute,
                    kth=assets - neighbor_count,
                    axis=1,
                )[:, -neighbor_count:]
                chosen_values = numpy.take_along_axis(
                    absolute,
                    chosen,
                    axis=1,
                )
                order = numpy.argsort(chosen_values, axis=1)
                chosen = numpy.take_along_axis(chosen, order, axis=1)
            neighbor_indices[start:stop] = chosen
            neighbor_correlations[start:stop] = numpy.take_along_axis(
                correlations,
                chosen,
                axis=1,
            )
        return standard_deviation, neighbor_indices, neighbor_correlations

    @staticmethod
    def _getStandardizedGrids(
        pcaGrid: ReturnsPCAGrid,
        conditionedState: _ConditionedRiskState,
    ) -> list[numpy.ndarray]:
        """Recover standardized state coordinates from simple-return grids."""
        dense_grid = conditionedState.returnsVolaGrid
        return [
            (
                numpy.log1p(
                    dense_grid.gridValues[
                        asset,
                        dense_grid.validStateMask[asset],
                        0,
                    ]
                )
                - pcaGrid.logReturnMean[asset]
            )
            / pcaGrid.logReturnScale[asset]
            for asset in range(len(dense_grid))
        ]

    @staticmethod
    def _getStatePairCoefficients(
        standardizedGrids: list[numpy.ndarray],
        scenarioCenter: numpy.ndarray,
        conditionalStd: numpy.ndarray,
        neighborIndices: numpy.ndarray,
        neighborCorrelations: numpy.ndarray,
    ) -> CorrelationFactors:
        """Vectorize nonzero compatibility coefficients by asset pair."""
        first_assets = []
        first_states = []
        second_assets = []
        second_states = []
        coefficients = []
        for asset_a, neighbors in enumerate(neighborIndices):
            for position, asset_b_value in enumerate(neighbors):
                asset_b = int(asset_b_value)
                if asset_b <= asset_a:
                    continue
                rho = float(neighborCorrelations[asset_a, position])
                if abs(rho) < 1e-6:
                    continue

                sigma_a = conditionalStd[asset_a]
                sigma_b = conditionalStd[asset_b]
                denominator = sigma_b**2 * max(1.0 - rho**2, 1e-12)
                residual_a = standardizedGrids[asset_a] - scenarioCenter[asset_a]
                residual_b = standardizedGrids[asset_b] - scenarioCenter[asset_b]
                expected_b = rho * sigma_b / sigma_a * residual_a
                pair_coefficients = (
                    residual_b[None, :] - expected_b[:, None]
                ) ** 2 / denominator
                nonzero_a, nonzero_b = numpy.nonzero(pair_coefficients)
                if len(nonzero_a) == 0:
                    continue
                first_assets.append(
                    numpy.full(len(nonzero_a), asset_a, dtype=numpy.int32)
                )
                first_states.append(nonzero_a.astype(numpy.int32, copy=False))
                second_assets.append(
                    numpy.full(len(nonzero_b), asset_b, dtype=numpy.int32)
                )
                second_states.append(nonzero_b.astype(numpy.int32, copy=False))
                coefficients.append(pair_coefficients[nonzero_a, nonzero_b])

        if not coefficients:
            return CorrelationFactors.empty()
        return CorrelationFactors(
            numpy.concatenate(first_assets),
            numpy.concatenate(first_states),
            numpy.concatenate(second_assets),
            numpy.concatenate(second_states),
            numpy.concatenate(coefficients),
        )
