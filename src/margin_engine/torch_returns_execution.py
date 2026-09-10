"""Own the lifetime of resident returns numerics behind a host outcome interface."""

from __future__ import annotations

from itertools import product
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from portfolio import Portfolio
    from risk_state_generator import RiskStateGenerationContext
    from .calculation_measurements import CalculationMeasurements
    from .numerical_execution_config import TorchNumericalExecutionConfig
import math

from margin_calculator import BQMMarginCalculator, BatchBQMExecutionPolicy, GreedyMarginCalculator
from .numerical_execution_config import validateResidentCollaborators
from margin_calculator.optimization.optimization_problem.qubo_problem import QUBOProblem
from margin_calculator.optimization.optimization_solver.bqm_solver.resource_plan import BQMResourcePlan
from margin_calculator.optimization.optimization_solver.bqm_solver.execution_memory import ExecutionMemoryTracker
from margin_calculator.optimization.optimization_solver.bqm_solver.torch_qubo import TorchQUBO
from margin_calculator.optimization.optimization_solver.bqm_solver.torch_execution import TorchExecution
from risk_state_generator.risk_state import CorrelationFactors

import numpy
from numerics.neighbor_selection import torchStrongestNeighbors

from margin_calculator.calculation_outcome import CalculationOutcome
from margin_calculator.state_aware_greedy_margin_calculator import StateAwareGreedyMarginCalculator
from risk_state_generator import ReturnsPCAGrid, ReturnsPCAKey, CorrelatedReturnsVolaGridRiskStateGenerator
from risk_state_generator.pca_backend import TorchPCABackend
from risk_state_generator.returns_vola_grid_risk_state_generator import ReturnsVolaGridRiskStateGenerator


@dataclass(frozen=True)
class _ConditionedGrid:
    returns: Any
    mask: Any
    center: Any
    nearest: Any
    inflation: Any
    sigma: Any
    means: Any
    scales: Any
    fallbackMask: Any


class TorchReturnsExecution:
    """Execute one returns calculation with invocation-owned Torch intermediates."""

    def __init__(self, config: TorchNumericalExecutionConfig,
                 generator: ReturnsVolaGridRiskStateGenerator,
                 calculator: BQMMarginCalculator | StateAwareGreedyMarginCalculator) -> None:
        if type(generator) not in (ReturnsVolaGridRiskStateGenerator, CorrelatedReturnsVolaGridRiskStateGenerator):
            raise ValueError("Torch numerical execution requires a returns-grid risk generator")
        if type(calculator) is BQMMarginCalculator:
            validateResidentCollaborators(calculator.bqmVisitor, calculator.executionPolicy, calculator.comparisonVisitor)
            if not isinstance(calculator.bqmSolver, TorchExecution):
                raise ValueError("Resident BQM execution requires a Torch solver")
            if len(calculator.bqmSolver.devices) > 1:
                raise ValueError("Resident numerical execution requires one solver device")
        elif type(calculator) not in (StateAwareGreedyMarginCalculator, GreedyMarginCalculator):
            raise ValueError("Unsupported calculator for resident numerical execution")
        self.config = config
        self.generator = generator
        self.calculator = calculator
        self.device = config.device
        if isinstance(calculator, BQMMarginCalculator):
            import torch
            solver_device = torch.device(calculator.bqmSolver.device)
            requested = solver_device if config.device == "auto" else torch.device(config.device)
            def identity(device):
                return device.type, (torch.cuda.current_device() if device.index is None else device.index) if device.type == "cuda" else None
            if identity(requested) != identity(solver_device):
                raise ValueError("numericalExecution and Torch solver must use the same device")
            self.device = str(requested)

    def calculate(self, context: RiskStateGenerationContext, portfolio: Portfolio,
                  measurements: CalculationMeasurements) -> CalculationOutcome:
        import torch

        generator = self.generator
        key = ReturnsPCAKey(context.dataRequest.instruments, generator.ew_window,
                            context.marginDate, generator.ew_lambda, generator.components)
        with torch.inference_mode(), torch.profiler.record_function("margin.resident"):
            def prepare():
                inputs = ReturnsPCAGrid.prepareInput(key, context.marketData)
                fit = TorchPCABackend(self.device, self.config.dtype).fitResident(
                    inputs.values, inputs.weights, key.components)
                return inputs, fit
            inputs, fit = measurements.measure("riskStateGenerationSeconds", prepare)
            device = fit.lambdas.device
            dtype = fit.lambdas.dtype
            positions = torch.tensor([float(portfolio.weights.get(item, 0)) for item in key.instruments],
                                     dtype=dtype, device=device)
            if not bool(torch.isfinite(positions).all()):
                raise ValueError("portfolio contains a non-finite weight")
            means = torch.tensor(inputs.logReturnMean, dtype=dtype, device=device)
            scales = torch.tensor(inputs.logReturnScale, dtype=dtype, device=device)
            diagnostics = {
                "mode": "torch_resident", "device": str(device), "dtype": str(dtype).removeprefix("torch."), "scenarioCount": 0,
                "residentFitBytes": sum(value.numel() * value.element_size() for value in vars(fit).values()),
                "fittedHostMaterializationBytes": 0, "sourceSnapshotBytes": 0,
                "peakBatchProblems": 0, "peakAdmittedWorkingBytes": 0,
            }
            grids = self._grids(fit, means, scales, key.instruments, measurements, diagnostics)
            if isinstance(self.calculator, BQMMarginCalculator):
                return self._bqm(grids, positions, key.instruments, diagnostics)
            worst = torch.zeros((), dtype=dtype, device=device)
            for grid in grids:
                returns, mask = grid.returns, grid.mask
                weighted = positions[:, None] * returns
                worst = torch.minimum(worst, weighted.masked_fill(~mask, torch.inf).amin(dim=1).sum())
            return CalculationOutcome(-float(worst), numericalDiagnostics=diagnostics)

    def _bqm(self, grids, positions, instruments, diagnostics):
        import torch

        calculator = self.calculator
        solver = calculator.bqmSolver
        policy = calculator.executionPolicy
        batch_size = policy.batchSize if isinstance(policy, BatchBQMExecutionPolicy) else 1
        budget = policy.maxBatchBytes if isinstance(policy, BatchBQMExecutionPolicy) else None
        multiplier = policy.memoryMultiplier if isinstance(policy, BatchBQMExecutionPolicy) else 3.
        batch, costs, retained = [], [], []
        topology_cache = {}
        worst, greedy = 0., 0.
        memory = ExecutionMemoryTracker(policy.memoryObserver if isinstance(policy, BatchBQMExecutionPolicy) else None)
        fixed_bytes = diagnostics["residentFitBytes"] + positions.numel() * 24

        def resource_plan(sizes, host_bytes=0):
            return BQMResourcePlan.create([sizes[0] + fixed_bytes, *sizes[1:]], 1, host_bytes)

        def solve():
            nonlocal worst
            plan = resource_plan(costs, sum(item.source.numericMemoryBytes for item in batch))
            diagnostics["peakBatchProblems"] = max(diagnostics["peakBatchProblems"], len(batch))
            diagnostics["peakAdmittedWorkingBytes"] = max(diagnostics["peakAdmittedWorkingBytes"], sum(plan.workerBytes))
            memory.update(active=plan.hostBytes)
            try:
                results = solver.solveResidentPlanned(batch, plan, calculator.solverParameters)
            finally:
                memory.update(retainedDelta=-plan.hostBytes, active=0)
            for result, weighted in zip(results, retained):
                # Repair/selection guarantees one-hot feasibility; use source
                # weighted returns, not penalty-contaminated QUBO energy.
                worst = min(worst, sum(float(value) for value, selected in zip(weighted, result.sample) if selected))
            batch.clear()
            costs.clear()
            retained.clear()

        solver.beginSeries()
        try:
            for grid in grids:
                returns, mask = grid.returns, grid.mask
                weighted = positions[:, None] * returns
                if calculator.comparisonVisitor is not None:
                    greedy = min(greedy, float(weighted.masked_fill(~mask, torch.inf).amin(dim=1).sum()))
                problem, source_returns = self._encode(weighted, grid, instruments, topology_cache)
                memory.update(retainedDelta=problem.source.numericMemoryBytes)
                diagnostics["sourceSnapshotBytes"] += problem.source.numericMemoryBytes
                cost = max(int(problem.source.numericMemoryBytes * multiplier),
                           solver.estimatedWorkingMemoryBytes(problem.source, calculator.solverParameters)) + problem.numericMemoryBytes
                if batch and not resource_plan(costs + [cost]).fits(budget):
                    solve()
                batch.append(problem)
                costs.append(cost)
                retained.append(source_returns)
                if len(batch) == batch_size:
                    solve()
            if batch:
                solve()
        finally:
            try:
                grids.close()
            finally:
                try:
                    memory.update(finished=True)
                finally:
                    solver.endSeries()
        comparisons = {} if calculator.comparisonVisitor is None else {"greedy": -greedy}
        return CalculationOutcome(-worst, comparisons, diagnostics)

    def _encode(self, weighted, grid, instruments, topology_cache):
        import torch

        penalty = self.calculator.modelParameters.get("lambdaOneHot", 1.)
        if not math.isfinite(penalty) or penalty < 0:
            raise ValueError("lambdaOneHot must be finite and nonnegative")
        mask = grid.mask
        counts = tuple(int(value) for value in mask.sum(dim=1).cpu().numpy())
        template = self.calculator.bqmVisitor.oneHotTopology(counts, penalty)
        device = weighted.device
        selected = weighted[mask]
        # The immutable host snapshot owns source scoring, deterministic seed
        # identity and repair. Float32 risk coefficients are widened before
        # host penalty arithmetic; source scoring uploads that precise snapshot.
        host_returns = selected.cpu().numpy().astype(numpy.float64, copy=True)
        linear = host_returns - penalty
        compatibility = self.calculator.modelParameters.get("lambdaCompat", .1) if isinstance(
            self.generator, CorrelatedReturnsVolaGridRiskStateGenerator) else 0.
        if not math.isfinite(compatibility) or compatibility < 0:
            raise ValueError("lambdaCompat must be finite and nonnegative")
        correlation, device_correlation = self._correlations(grid)
        topology_key = counts, penalty
        if topology_key not in topology_cache:
            # Keep only the current shape, bounded to this calculation's lifetime.
            topology_cache.clear()
            topology_cache[topology_key] = (
                torch.tensor(template.heads, dtype=torch.int64, device=device),
                torch.tensor(template.tails, dtype=torch.int64, device=device),
                torch.full((len(template.biases),), 2 * penalty, dtype=weighted.dtype, device=device))
        heads, tails, biases = topology_cache[topology_key]
        host_heads, host_tails, host_biases = template.heads, template.tails, template.biases
        if len(correlation) and compatibility:
            first_assets, first_states, second_assets, second_states, coefficients = device_correlation
            # Source precision owns the cutoff so float32 rounding cannot change
            # topology between resident dynamics and authoritative host scoring.
            normalized = self.calculator.bqmVisitor.normalizedCorrelationCoefficients(correlation)
            retained = normalized != 0.0
            device_retained = torch.tensor(retained, device=device)
            first_assets, first_states, second_assets, second_states = (
                values[device_retained]
                for values in (first_assets, first_states, second_assets, second_states))
            coefficients = torch.tensor(normalized[retained], dtype=coefficients.dtype, device=device)
            offsets = torch.tensor(template.offsets, device=device)
            heads = torch.cat((heads, offsets[first_assets] + first_states))
            tails = torch.cat((tails, offsets[second_assets] + second_states))
            biases = torch.cat((biases, coefficients * compatibility))
            host_heads = numpy.concatenate((host_heads, template.offsets[correlation.firstAssets[retained]] + correlation.firstStates[retained]))
            host_tails = numpy.concatenate((host_tails, template.offsets[correlation.secondAssets[retained]] + correlation.secondStates[retained]))
            host_biases = numpy.concatenate((host_biases, normalized[retained] * compatibility))
        source = QUBOProblem(linear, host_heads, host_tails, host_biases,
            offset=template.offset, groupOffsets=template.offsets,
            seedOffset=self.calculator.bqmVisitor.scenarioSeedOffset(instruments, linear, correlation, penalty, compatibility))
        return TorchQUBO(source, selected - penalty, heads, tails, biases), host_returns

    def _correlations(self, grid):
        """Construct the symmetric nomination union in bounded device blocks."""
        import torch

        generator = self.generator
        if not isinstance(generator, CorrelatedReturnsVolaGridRiskStateGenerator):
            return CorrelationFactors.empty(), None
        samples = grid.nearest * grid.inflation
        observations, assets = samples.shape
        neighbors = min(generator.topKNeighbors, max(assets - 1, 0))
        if not neighbors:
            return CorrelationFactors.empty(), None
        scale = max(observations - 1, 1)
        centered = samples - samples.mean(dim=0) if observations > 1 else torch.zeros_like(samples)
        variance = centered.square().sum(dim=0) / scale
        variance = torch.where(torch.isfinite(variance) & (variance > 1e-12), variance, grid.sigma.square().clamp_min(1e-12))
        std = variance.sqrt()
        normalized = centered / std
        pair_keys, pair_values = [], []
        rows = max(1, min(assets, generator.correlationBlockBytes // (64 * assets)))
        with torch.profiler.record_function("margin.correlations"):
            for start in range(0, assets, rows):
                stop = min(start + rows, assets)
                correlations = torch.nan_to_num(normalized[:, start:stop].T @ normalized / scale,
                                                nan=0., posinf=0., neginf=0.).clamp(-.999, .999)
                absolute = correlations.abs()
                local = torch.arange(stop - start, device=samples.device)
                asset = torch.arange(start, stop, device=samples.device)
                absolute[local, asset] = -torch.inf
                chosen = torchStrongestNeighbors(torch, absolute, neighbors)
                values = correlations.gather(1, chosen)
                nominated = asset[:, None].expand_as(chosen)
                valid = values.abs() >= 1e-6
                pair_keys.append((torch.minimum(nominated, chosen) * assets + torch.maximum(nominated, chosen))[valid])
                pair_values.append(values[valid])
            keys = torch.cat(pair_keys)
            if not len(keys):
                return CorrelationFactors.empty(), None
            order = torch.argsort(keys, stable=True)
            keys, values = keys[order], torch.cat(pair_values)[order]
            unique, inverse, count = torch.unique_consecutive(keys, return_inverse=True, return_counts=True)
            rho = torch.zeros(len(unique), dtype=samples.dtype, device=samples.device).scatter_add_(0, inverse, values) / count
            first, second = unique // assets, unique % assets
            coordinates = (torch.log1p(grid.returns) - grid.means[:, None]) / grid.scales[:, None]
            residuals = (coordinates - grid.center[:, None]) / std[:, None]
            ranks = grid.mask.long().cumsum(dim=1) - 1
            output = [[], [], [], [], []]
            bins = grid.mask.shape[1]
            pairs_per_block = max(1, generator.correlationBlockBytes // max(80 * bins * bins, 1))
            for start in range(0, len(unique), pairs_per_block):
                stop = min(start + pairs_per_block, len(unique))
                a, b, r = first[start:stop], second[start:stop], rho[start:stop].clamp(-.999, .999)
                za, zb = residuals[a, :, None], residuals[b, None, :]
                coefficients = ((za.square() - 2 * r[:, None, None] * za * zb + zb.square()) /
                                (1 - r.square()).clamp_min(1e-12)[:, None, None]).clamp_min(0)
                valid = grid.mask[a, :, None] & grid.mask[b, None, :] & (coefficients != 0)
                pair, bin_a, bin_b = torch.nonzero(valid, as_tuple=True)
                for target, value in zip(output, (a[pair], ranks[a[pair], bin_a], b[pair],
                                                  ranks[b[pair], bin_b], coefficients[valid])):
                    target.append(value)
            arrays = tuple(torch.cat(parts) for parts in output)
            host = [value.cpu().numpy().astype(numpy.int32 if index < 4 else numpy.float64, copy=False)
                    for index, value in enumerate(arrays)]
            return CorrelationFactors(*host), arrays

    def _grids(self, fit, means, scales, instruments, measurements, diagnostics):
        import torch

        generator = self.generator
        counts = generator.scenariosPerComponents
        if len(counts) != generator.components or any(count % 2 == 0 for count in counts):
            raise ValueError("scenariosPerComponents must match components and contain positive odd integers")
        host_eigenvalues = fit.lambdas.cpu().numpy()
        diagnostics["fittedHostMaterializationBytes"] += host_eigenvalues.nbytes
        axes = generator._buildComponentGrids(host_eigenvalues, counts, generator.tailDensityGamma)
        fallback_used = torch.zeros(len(instruments), dtype=torch.bool, device=means.device)
        try:
            for point in product(*(axis.tolist() for axis in axes)):
                diagnostics["scenarioCount"] += 1
                grid = measurements.measure("riskStateGenerationSeconds", lambda: self._condition(
                    fit, means, scales, instruments, point))
                fallback_used |= grid.fallbackMask
                yield grid
        finally:
            diagnostics["fallbackAssetCount"] = int(fallback_used.sum())

    def _condition(self, fit, means, scales, instruments, point):
        import torch

        generator = self.generator
        with torch.profiler.record_function("margin.condition"):
            scenario = torch.tensor(point, dtype=means.dtype, device=means.device)
            center = fit.pcaMean + scenario @ fit.loadings
            distances = torch.linalg.vector_norm((fit.factors - scenario) / fit.lambdas.clamp_min(1e-12).sqrt(), dim=1)
            count = generator.nNearest or min(100, len(distances), generator.ew_window)
            if count > len(distances):
                raise ValueError("nNearest exceeds available residual observations")
            nearest_indices = torch.argsort(distances, stable=True)[:count].sort().values
            nearest = fit.residuals if count == len(distances) else fit.residuals[nearest_indices]
            selected_distances = distances if count == len(distances) else distances[nearest_indices]
            inflation = (1 + generator.distanceInflationAlpha * selected_distances.mean().pow(
                generator.distanceInflationPower)).clamp_max(generator.maxInflationFactor)
            local = nearest.std(dim=0, correction=1) if count > 1 else torch.full_like(means, torch.nan)
            sigma = torch.where(torch.isfinite(local) & (local > 1e-12), local, fit.residualScale)
            sigma = torch.where(torch.isfinite(sigma) & (sigma > 1e-12), sigma, torch.ones_like(sigma)) * inflation
            edges = torch.tensor(numpy.linspace(-1., 1., generator.nZBins + 1), dtype=means.dtype, device=means.device)
            centers = (edges[:-1] + edges[1:]) * .5
            bounds = center[:, None] + fit.maxAbsoluteZ[:, None] * edges
            z = center[:, None] + fit.maxAbsoluteZ[:, None] * centers
            width = generator.residualSigmaRange * sigma
            mask = (bounds[:, :-1] >= center[:, None] - width[:, None]) & (bounds[:, 1:] <= center[:, None] + width[:, None])
            empty = ~mask.any(dim=1)
            if not generator.allowEmptyBinFallback and bool(empty.any()):
                asset = int(torch.nonzero(empty)[0, 0])
                raise ValueError(f"No valid return bins for {instruments[asset]}")
            nearest_bin = (z - center[:, None]).abs().argmin(dim=1)
            mask[torch.arange(len(instruments), device=means.device), nearest_bin] |= empty
            returns = torch.expm1(means[:, None] + scales[:, None] * z)
            if not bool(torch.isfinite(returns).all()) or not bool(torch.isfinite(sigma).all()):
                raise ValueError("risk-state values must be finite")
            return _ConditionedGrid(returns, mask, center, nearest, inflation, sigma, means, scales, empty)
