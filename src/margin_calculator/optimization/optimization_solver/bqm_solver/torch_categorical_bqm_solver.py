"""Feasible categorical annealing over complete disjoint one-hot groups."""

from dataclasses import dataclass
import math

import numpy
from scipy.sparse import coo_matrix

from margin_calculator.optimization.optimization_result import BQMOptimizationResult
from .bqm_solver_factory import BQMSolverFactory
from .torch_candidates import TorchCandidateAccumulator
from .torch_execution import TorchExecution, _MAX_TORCH_SEED, _RUN_SEED_STRIDE


@dataclass(frozen=True)
class _CategoricalModel:
    groups: tuple
    linear: numpy.ndarray
    adjacency: object
    colors: tuple

    @classmethod
    def fromProblem(cls, problem):
        groups = tuple(tuple(g) for g in problem.iterOneHotGroups())
        if not groups or sum(map(len, groups)) != problem.variableCount:
            raise ValueError("Torch categorical solver requires one-hot groups covering every variable")
        owner = numpy.empty(problem.variableCount, dtype=numpy.int64)
        for index, group in enumerate(groups):
            owner[list(group)] = index
        heads, tails, biases = problem.quadraticHeads, problem.quadraticTails, problem.quadraticBiases
        linear = problem.linear.copy()
        diagonal = heads == tails
        numpy.add.at(linear, heads[diagonal], biases[diagonal])
        # Distinct variables in one group can never both be active. This removes
        # one-hot penalty cliques and any other identically zero within-group terms.
        cross = owner[heads] != owner[tails]
        adjacency = coo_matrix((numpy.concatenate((biases[cross], biases[cross])),
            (numpy.concatenate((heads[cross], tails[cross])), numpy.concatenate((tails[cross], heads[cross])))),
            shape=(problem.variableCount, problem.variableCount)).tocsr()
        adjacency.sum_duplicates()
        adjacency.eliminate_zeros()
        # Common group shifts are constant on the feasible space; remove them in
        # float64 before casting so large one-hot penalties do not dilute dynamics.
        for group in groups:
            linear[list(group)] -= linear[list(group)].min()
        neighbors = [set() for _ in groups]
        row, column = adjacency.nonzero()
        for a, b in zip(owner[row], owner[column]):
            neighbors[a].add(b)
        labels = []
        for index in range(len(groups)):
            occupied = {labels[n] for n in neighbors[index] if n < index}
            color = 0
            while color in occupied:
                color += 1
            labels.append(color)
        colors = tuple(tuple(i for i, label in enumerate(labels) if label == color)
                       for color in range(max(labels) + 1))
        return cls(groups, linear, adjacency, colors)


class TorchCategoricalBQMSolver(TorchExecution):
    """Graph-colored heat-bath annealing; every candidate is one-hot by construction.

    Trajectories run together on one device; problems run sequentially within each
    admitted device shard. This is a categorical solver, not SBM or SVL dynamics.
    """

    _defaults = {"steps": 64, "runs": 16, "seed": 1, "dtype": "float32",
        "run_batch_size": None, "energy_chunk_size": 1_000_000,
        "temperature_start": 1., "temperature_end": .01, "greedy_sweeps": 4,
        "noise_chunk_size": 16}

    @classmethod
    def _getParameters(cls, solverParameters):
        supplied = dict(solverParameters or {})
        unknown = supplied.keys() - cls._defaults.keys()
        if unknown:
            raise ValueError(f"Unknown Torch categorical parameters: {sorted(unknown)}")
        values = cls._defaults | supplied
        for key in ("steps", "runs", "seed", "energy_chunk_size", "greedy_sweeps", "noise_chunk_size"):
            value = values[key]
            if isinstance(value, bool) or not isinstance(value, (int, numpy.integer)):
                raise TypeError(f"Torch categorical {key} must be an integer")
            values[key] = int(value)
        if min(values[k] for k in ("steps", "runs", "energy_chunk_size", "noise_chunk_size")) <= 0:
            raise ValueError("Torch categorical steps, runs and chunk sizes must be positive")
        if values["greedy_sweeps"] < 0:
            raise ValueError("Torch categorical greedy_sweeps must be nonnegative")
        if not 0 <= values["seed"] < _MAX_TORCH_SEED:
            raise ValueError("Torch categorical seed must be in [0, 2**63 - 1)")
        if values["dtype"] not in ("float32", "float64"):
            raise ValueError("Torch categorical dtype must be float32 or float64")
        for key in ("temperature_start", "temperature_end"):
            values[key] = float(values[key])
            if not math.isfinite(values[key]) or values[key] < 0:
                raise ValueError(f"Torch categorical {key} must be finite and nonnegative")
        if values["temperature_end"] > values["temperature_start"]:
            raise ValueError("Torch categorical temperature_end must not exceed temperature_start")
        width = values["run_batch_size"]
        if width is not None and (isinstance(width, bool) or not isinstance(width, (int, numpy.integer)) or width <= 0):
            raise ValueError("Torch categorical run_batch_size must be a positive integer or None")
        return values

    def estimatedWorkingMemoryBytes(self, problem, solverParameters=None):
        parameters = self._getParameters(solverParameters)
        groups = tuple(problem.iterOneHotGroups())
        if not groups or sum(map(len, groups)) != problem.variableCount:
            raise ValueError("Torch categorical solver requires one-hot groups covering every variable")
        width = min(parameters["run_batch_size"] or parameters["runs"], parameters["runs"])
        # Includes ragged-to-padded costs, sparse canonicalization, RNG chunks,
        # original float64 scoring, and both host/device coefficient copies.
        padding = len(groups) * max(map(len, groups))
        return super().estimatedWorkingMemoryBytes(problem, parameters) + padding * (32 * width + 16)

    def _solveBatch(self, problems, parameters):
        torch = self._torch()
        device = torch.device(self.device)
        dtype = getattr(torch, parameters["dtype"])
        results = []
        with torch.inference_mode():
            for problem in problems:
                model = _CategoricalModel.fromProblem(problem)
                matrix = torch.sparse_csr_tensor(torch.tensor(model.adjacency.indptr, device=device),
                    torch.tensor(model.adjacency.indices, device=device),
                    torch.tensor(model.adjacency.data, dtype=dtype, device=device),
                    size=model.adjacency.shape, check_invariants=False)
                linear = torch.tensor(model.linear, dtype=dtype, device=device)[:, None]
                if not bool(torch.isfinite(linear).all()) or not bool(torch.isfinite(matrix.values()).all()):
                    raise ValueError("Categorical coefficients must remain finite in the selected dtype")
                groups = model.groups
                tables = []
                for color in model.colors:
                    table = numpy.full((len(color), max(len(groups[g]) for g in color)), problem.variableCount, dtype=numpy.int64)
                    for row, g in enumerate(color):
                        table[row, :len(groups[g])] = groups[g]
                    tables.append((torch.tensor(color, device=device), torch.tensor(table, device=device)))
                group_starts = torch.tensor([g[0] for g in groups], device=device)
                accumulator = TorchCandidateAccumulator(torch, problem, device,
                    parameters["energy_chunk_size"], self._selectBestCandidates)
                batch_size = min(parameters["run_batch_size"] or parameters["runs"], parameters["runs"])
                for start in range(0, parameters["runs"], batch_size):
                    width = min(batch_size, parameters["runs"] - start)
                    samples = self._runCategorical(torch, model, matrix, linear, tables,
                        group_starts, problem.seedOffset, parameters, start, width)
                    accumulator.add(samples.T)
                results.append(BQMOptimizationResult(*accumulator.result()))
        return results

    def solveResidentPlanned(self, problems, plan, solverParameters=None):
        """Compile categorical topology from authoritative host source snapshots."""
        if not problems:
            return []
        if len(self.devices) > 1 or plan.shards != ((0, len(problems)),):
            raise ValueError("Resident categorical plan must cover one ordered device shard")
        # Resident Ising coefficients contain penalty cliques that are unnecessary
        # here. The host snapshot is already required for exact scoring and seeds.
        return self.solvePlanned([problem.source for problem in problems], plan, solverParameters)

    @staticmethod
    def _runCategorical(torch, model, matrix, linear, tables, groupStarts, seedOffset, parameters, runStart, width):
        device, dtype = linear.device, linear.dtype
        variables = len(linear)
        choices = groupStarts[:, None].expand(-1, width).clone()
        generators = []
        for run in range(width):
            seed = (parameters["seed"] + int(seedOffset) + _RUN_SEED_STRIDE * (runStart + run)) % _MAX_TORCH_SEED
            generator = torch.Generator(device=device).manual_seed(seed)
            # A group table handles noncontiguous and unequal-size groups too.
            u = torch.rand(len(model.groups), generator=generator, device=device, dtype=dtype)
            for color, table in tables:
                lengths = (table != variables).sum(dim=1)
                index = (u[color] * lengths).long()
                choices[color, run] = table.gather(1, index[:, None]).squeeze(1)
            generators.append(generator)
        samples = torch.zeros((variables, width), dtype=dtype, device=device)
        samples.scatter_(0, choices, 1.)
        field = torch.empty((variables + 1, width), dtype=dtype, device=device)
        chunk_size = min(parameters["noise_chunk_size"], parameters["steps"])
        noise = torch.empty((chunk_size, variables + 1, width), dtype=dtype, device=device) if parameters["temperature_start"] else None
        for step in range(parameters["steps"] + parameters["greedy_sweeps"]):
            fraction = step / max(parameters["steps"] - 1, 1)
            temperature = (parameters["temperature_start"] * max(0., 1 - fraction)
                           + parameters["temperature_end"] * min(1., fraction)) if step < parameters["steps"] else 0.
            if temperature and step % chunk_size == 0:
                for run, generator in enumerate(generators):
                    noise[:, :variables, run] = torch.rand((chunk_size, variables), generator=generator, device=device, dtype=dtype)
                noise[:, :variables].clamp_(min=torch.finfo(dtype).tiny, max=1 - torch.finfo(dtype).eps)
                # log(-log(U)) is negative Gumbel noise for a minimum-cost draw.
                noise[:, :variables].log_().neg_().log_()
                noise[:, variables] = 0.
            for color, table in tables:
                torch.addmm(field[:variables], matrix, samples, beta=0, out=field[:variables])
                field[:variables].add_(linear)
                field[variables] = torch.inf
                costs = field[table]
                if temperature:
                    costs = costs + temperature * noise[step % chunk_size][table]
                best = costs.argmin(dim=1)
                selected = table.gather(1, best).reshape(len(color), width)
                if not temperature:
                    # Preserve the current category on ties; deterministic descent
                    # then cannot oscillate between equal-energy categories.
                    previous = choices[color]
                    best_cost = field.gather(0, selected)
                    old_cost = field.gather(0, previous)
                    selected = torch.where(best_cost < old_cost - 1e-12, selected, previous)
                choices[color] = selected
                # Category indices are authoritative; rebuild the one-hot view for
                # sparse conditional costs. Every color update stays feasible.
                samples.zero_().scatter_(0, choices, 1.)
        return samples.to(torch.uint8)


BQMSolverFactory.registerSolver("torch_categorical", TorchCategoricalBQMSolver)
