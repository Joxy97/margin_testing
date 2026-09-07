"""Deterministic filtered-historical-simulation/EVT risk-state generator."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
import math

import numpy
import pandas

from download_unit import DataRequest, Period
from portfolio import Portfolio

from .fhs_evt import (
    ConditionalFilter,
    ConditionalFilterResult,
    DependenceCells,
    DeterministicRule,
    EVTMarginalFitter,
    FHSScenario,
    ScenarioReducer,
    ScenarioType,
    ScenarioValidationReport,
    SemiparametricMarginal,
    buildJointTailStresses,
    constructDependenceCells,
    generateProbabilityScenarios,
    prepareLogReturnHistory,
    validateScenarioSet,
)
from .risk_state import DenseReturnsVolaGrid, FHSEVTRiskState
from .risk_state_generation_context import RiskStateGenerationContext
from .risk_state_generator import RiskStateGenerator


class FHSEVTRiskStateGenerator(RiskStateGenerator):
    """Generate exactly 105 joint FHS-EVT scenarios by default.

    Close-price instruments are modelled as synchronized log returns. The
    reduced scenario rows are converted to simple returns at the existing
    portfolio/QUBO boundary.
    """

    def __init__(
        self,
        historyDays: int = 1825,
        minimumObservations: int = 252,
        meanModel: str = "constant",
        arOrder: int = 1,
        varianceModel: str = "gjr_garch",
        burnIn: int = 50,
        ewmaLambda: float = 0.94,
        persistenceBuffer: float = 0.005,
        optimizerMaxIterations: int = 500,
        rowWeightDecay: float = 1.0,
        tailMassCandidates: tuple[float, ...] = (0.025, 0.05, 0.075, 0.10),
        minimumTailObservations: int = 20,
        evtShapeLowerBound: float = -0.45,
        evtShapeUpperBound: float = 0.45,
        evtQuadraturePoints: int = 512,
        thresholdStabilityTolerance: float = 0.20,
        integrationNodes: int = 1,
        endpointProbability: float = 1e-10,
        stressSigmaLevels: tuple[int, ...] = (3, 4, 5),
        targetScenarios: int = 105,
        reduceScenarios: bool = True,
        protectProbabilityExtremes: bool = True,
        localSwapPasses: int = 1,
        recalibrationIntervalDays: int = 0,
        calibrationWorkers: int = 1,
        modelVersion: str = "fhs-evt-v1",
    ) -> None:
        for name, value, lower in (
            ("historyDays", historyDays, 1),
            ("minimumObservations", minimumObservations, 3),
            ("integrationNodes", integrationNodes, 1),
            ("targetScenarios", targetScenarios, 1),
            ("recalibrationIntervalDays", recalibrationIntervalDays, 0),
            ("calibrationWorkers", calibrationWorkers, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < lower:
                raise ValueError(f"{name} must be an integer of at least {lower}")
        if not math.isfinite(rowWeightDecay) or not 0.0 < rowWeightDecay <= 1.0:
            raise ValueError("rowWeightDecay must lie in (0, 1]")
        if (
            not math.isfinite(endpointProbability)
            or not 0.0 < endpointProbability < 0.5
        ):
            raise ValueError("endpointProbability must lie in (0, .5)")
        if not modelVersion:
            raise ValueError("modelVersion must not be empty")
        if not isinstance(reduceScenarios, bool):
            raise TypeError("reduceScenarios must be a bool")
        if reduceScenarios and targetScenarios <= 2 * len(stressSigmaLevels):
            raise ValueError(
                "targetScenarios must leave room for a probability scenario "
                "after all protected joint-tail stresses"
            )
        self.historyDays = historyDays
        self.minimumObservations = minimumObservations
        self.rowWeightDecay = float(rowWeightDecay)
        self.integrationNodes = integrationNodes
        self.endpointProbability = float(endpointProbability)
        self.stressSigmaLevels = tuple(stressSigmaLevels)
        self.targetScenarios = targetScenarios
        self.reduceScenarios = reduceScenarios
        self.recalibrationIntervalDays = recalibrationIntervalDays
        self.calibrationWorkers = calibrationWorkers
        self.modelVersion = str(modelVersion)
        self.conditionalFilter = ConditionalFilter(
            meanModel=meanModel,
            arOrder=arOrder,
            varianceModel=varianceModel,
            burnIn=burnIn,
            ewmaLambda=ewmaLambda,
            persistenceBuffer=persistenceBuffer,
            optimizerMaxIterations=optimizerMaxIterations,
        )
        self.marginalFitter = EVTMarginalFitter(
            tailMassCandidates=tuple(tailMassCandidates),
            minimumTailObservations=minimumTailObservations,
            shapeLowerBound=evtShapeLowerBound,
            shapeUpperBound=evtShapeUpperBound,
            quadraturePoints=evtQuadraturePoints,
            thresholdStabilityTolerance=thresholdStabilityTolerance,
        )
        self.reducer = ScenarioReducer(
            targetScenarios=targetScenarios,
            protectProbabilityExtremes=protectProbabilityExtremes,
            localSwapPasses=localSwapPasses,
        )
        self.lastFilterResults: tuple[ConditionalFilterResult, ...] = ()
        self.lastMarginals: tuple[SemiparametricMarginal, ...] = ()
        self.lastDependenceCells: DependenceCells | None = None
        self.lastValidationReport: ScenarioValidationReport | None = None
        self.lastRetainedDates: tuple[date, ...] = ()
        self._calibrationDate: date | None = None
        self._calibrationInstruments: tuple[str, ...] = ()
        self._calibratedFilterResults: tuple[ConditionalFilterResult, ...] = ()
        self._calibratedMarginals: tuple[SemiparametricMarginal, ...] = ()
        self._calibratedDependence: DependenceCells | None = None
        self._calibratedRetainedDates: tuple[date, ...] = ()

    def createDataRequest(
        self,
        portfolio: Portfolio,
        marginDate: date,
    ) -> DataRequest:
        """Request the governed calendar history through the margin date."""
        return DataRequest(
            instruments=portfolio.instruments,
            start_date=marginDate - timedelta(days=self.historyDays),
            end_date=marginDate,
            data_type="closePrices",
            period=Period.ONE_DAY,
        )

    def getRiskStates(
        self,
        context: RiskStateGenerationContext,
    ) -> Iterator[FHSEVTRiskState]:
        """Fit, generate, reduce, validate, and yield joint scenario rows."""
        instruments = tuple(context.dataRequest.instruments)
        history = prepareLogReturnHistory(
            context.marketData,
            instruments,
            context.marginDate,
            self.minimumObservations,
        )
        if self._requiresCalibration(instruments, context.marginDate):
            filters = self._fitFilters(history.changes)
            retained_start = max(item.validStart for item in filters)
            residuals = numpy.column_stack(
                tuple(
                    item.standardizedResiduals[retained_start:]
                    for item in filters
                )
            )
            retained_dates = history.dates[retained_start:]
            retained_ids = history.rowIds[retained_start:]
            weights = self._rowWeights(len(residuals))
            marginals = self._fitMarginals(residuals, weights)
            dependence = constructDependenceCells(
                residuals,
                weights,
                retained_dates,
                retained_ids,
            )
            self._calibrationDate = context.marginDate
            self._calibrationInstruments = instruments
            self._calibratedFilterResults = filters
            self._calibratedMarginals = marginals
            self._calibratedDependence = dependence
            self._calibratedRetainedDates = retained_dates
        else:
            filters = tuple(
                self.conditionalFilter.applyCalibrated(
                    history.changes[:, factor],
                    self._calibratedFilterResults[factor],
                )
                for factor in range(history.changes.shape[1])
            )
            marginals = self._calibratedMarginals
            dependence = self._calibratedDependence
            retained_dates = self._calibratedRetainedDates
            if dependence is None:  # pragma: no cover - guarded by calibration
                raise RuntimeError("FHS-EVT calibration snapshot is incomplete")
        mean_forecast = numpy.asarray(
            [item.meanForecast for item in filters], dtype=numpy.float64
        )
        volatility_forecast = numpy.asarray(
            [item.volatilityForecast for item in filters], dtype=numpy.float64
        )
        rule = DeterministicRule.build(len(instruments), self.integrationNodes)
        probability = generateProbabilityScenarios(
            dependence,
            marginals,
            rule,
            mean_forecast,
            volatility_forecast,
            self.endpointProbability,
            self.modelVersion,
        )
        lower_crisis_owner = self._crisisOwner(probability, lower=True)
        upper_crisis_owner = self._crisisOwner(probability, lower=False)
        stresses = buildJointTailStresses(
            marginals,
            mean_forecast,
            volatility_forecast,
            numpy.vstack(
                (
                    lower_crisis_owner.innovations,
                    upper_crisis_owner.innovations,
                )
            ),
            (
                lower_crisis_owner.ownerRowId or "unknown",
                upper_crisis_owner.ownerRowId or "unknown",
            ),
            self.stressSigmaLevels,
            self.modelVersion,
        )
        published = (
            self.reducer.reduce(probability, stresses)
            if self.reduceScenarios
            else probability
            + tuple(sorted(stresses, key=lambda item: item.scenarioId))
        )
        validation = validateScenarioSet(
            published,
            self.targetScenarios if self.reduceScenarios else None,
        )

        self.lastFilterResults = filters
        self.lastMarginals = marginals
        self.lastDependenceCells = dependence
        self.lastValidationReport = validation
        self.lastRetainedDates = retained_dates
        for scenario in published:
            simple_returns = numpy.expm1(scenario.factorChanges)
            if not numpy.isfinite(simple_returns).all():
                raise ValueError(
                    f"scenario {scenario.scenarioId} produces non-finite simple returns"
                )
            values = numpy.stack((simple_returns, volatility_forecast), axis=1)
            dense = DenseReturnsVolaGrid(
                instruments,
                values[:, None, :],
                numpy.ones((len(instruments), 1), dtype=bool),
            )
            yield FHSEVTRiskState(
                returnsVolaGrid=dense,
                scenarioId=scenario.scenarioId,
                scenarioType=scenario.scenarioType,
                probability=scenario.probability,
                factorChanges=scenario.factorChanges,
                protected=scenario.protected,
                protectionReason=scenario.protectionReason,
                ownerRowId=scenario.ownerRowId,
                nodeId=scenario.nodeId,
                stressFamily=scenario.stressFamily,
                sigmaLabel=scenario.sigmaLabel,
                assignedParentMass=scenario.assignedParentMass,
            )

    @staticmethod
    def _crisisOwner(
        scenarios: tuple[FHSScenario, ...],
        lower: bool,
    ) -> FHSScenario:
        """Select a stable broad lower/upper historical owner direction."""
        if not scenarios:
            raise ValueError("probability scenarios must not be empty")
        sign = 1.0 if lower else -1.0
        return min(
            scenarios,
            key=lambda item: (
                sign * float(numpy.mean(item.innovations)),
                item.scenarioId,
            ),
        )

    def _requiresCalibration(
        self,
        instruments: tuple[str, ...],
        currentDate: date,
    ) -> bool:
        if (
            self._calibrationDate is None
            or instruments != self._calibrationInstruments
            or currentDate < self._calibrationDate
        ):
            return True
        if self.recalibrationIntervalDays == 0:
            return True
        return (
            currentDate - self._calibrationDate
        ).days >= self.recalibrationIntervalDays

    def _fitFilters(
        self,
        changes: numpy.ndarray,
    ) -> tuple[ConditionalFilterResult, ...]:
        columns = tuple(
            changes[:, factor] for factor in range(changes.shape[1])
        )
        if self.calibrationWorkers == 1:
            return tuple(self.conditionalFilter.fit(column) for column in columns)
        with ThreadPoolExecutor(max_workers=self.calibrationWorkers) as executor:
            return tuple(executor.map(self.conditionalFilter.fit, columns))

    def _fitMarginals(
        self,
        residuals: numpy.ndarray,
        weights: numpy.ndarray,
    ) -> tuple[SemiparametricMarginal, ...]:
        columns = tuple(
            residuals[:, factor] for factor in range(residuals.shape[1])
        )
        if self.calibrationWorkers == 1:
            return tuple(
                self.marginalFitter.fit(column, weights) for column in columns
            )
        with ThreadPoolExecutor(max_workers=self.calibrationWorkers) as executor:
            return tuple(
                executor.map(
                    lambda column: self.marginalFitter.fit(column, weights),
                    columns,
                )
            )

    def _prepareMarketData(
        self,
        data: pandas.DataFrame,
        instruments: tuple[str, ...],
        marginDate: date,
    ) -> tuple[numpy.ndarray, tuple[date, ...], tuple[str, ...]]:
        """Compatibility wrapper around the dedicated market-data stage."""
        history = prepareLogReturnHistory(
            data,
            instruments,
            marginDate,
            self.minimumObservations,
        )
        return history.changes, history.dates, history.rowIds

    def _rowWeights(self, observations: int) -> numpy.ndarray:
        if observations <= 0:
            raise ValueError("no standardized residuals remain after burn-in")
        weights = self.rowWeightDecay ** numpy.arange(
            observations - 1,
            -1,
            -1,
            dtype=numpy.float64,
        )
        total = float(weights.sum())
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("rowWeightDecay underflowed the retained history")
        return weights / total
