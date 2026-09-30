"""Categorical flow batching/validation and reusable CUDA graph capture."""
from collections.abc import Mapping
from numbers import Integral, Real
from threading import Lock
from typing import Any
import math
import numpy
from .torch_execution import TorchExecution
_CAPTURE_LOCK = Lock()

class GraphCaptureWorkspace:
    def capture(self, kappas):
        """Capture a fixed step block without advancing the visible trajectory."""
        torch = self.torch
        with _CAPTURE_LOCK, torch.cuda.device(self.theta.device):
            saved = self.theta.clone()
            current = torch.cuda.current_stream(self.theta.device)
            stream = torch.cuda.Stream(device=self.theta.device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self.advance(kappas[0], kappas[1])
            current.wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream, capture_error_mode='thread_local'):
                for step in range(len(kappas) - 1):
                    self.advance(kappas[step], kappas[step + 1])
            current.wait_stream(stream)
            self.theta.copy_(saved)
        return graph

class CategoricalFlowExecution(TorchExecution):
    residentMode = "source_snapshot"
    @staticmethod
    def _getParameters(solverParameters: Mapping[str, Any] | None = None) -> dict[str, Any]:
        parameters = dict(
            steps=1000, runs=64, time_step=0.05, mobility=1.0,
            route_strength=1.0, gamma=0.0, kappa_initial=-1.0,
            kappa_final=2.0, schedule_exponent=1.0, integrator='euler',
            candidate_interval=25, candidate_batch_size=128,
            matrix_format='sparse', sparse_threshold=0.15,
            max_dense_variables=10000, max_variables=100000,
            dtype='float32', seed=0, run_batch_size=16,
            energy_chunk_size=8192, cuda_graph=True, graph_steps=25,
            deduplicate_candidates=True,
        )
        supplied = dict(solverParameters or {})
        # Translate upstream work-budget names at this solver boundary only.
        for alias, canonical in (('agents', 'runs'), ('max_steps', 'steps')):
            if alias in supplied:
                if canonical in supplied:
                    raise ValueError(f'Transverse route cannot define both {alias} and {canonical}')
                supplied[canonical] = supplied.pop(alias)
        unknown = supplied.keys() - parameters.keys()
        if unknown:
            raise ValueError(f'Unknown transverse-route parameters: {sorted(unknown)}')
        parameters.update(supplied)
        for name in ('steps', 'runs', 'candidate_interval', 'candidate_batch_size',
                     'energy_chunk_size', 'graph_steps', 'max_dense_variables', 'max_variables',
                     'seed', 'run_batch_size'):
            value = parameters[name]
            if name == 'run_batch_size' and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f'Categorical flow {name} must be an integer')
            if value < (0 if name == 'seed' else 1):
                raise ValueError(f'Categorical flow {name} is out of range')
            parameters[name] = int(value)
        for name in ('time_step', 'mobility', 'route_strength', 'gamma',
                     'kappa_initial', 'kappa_final', 'schedule_exponent', 'sparse_threshold'):
            value = parameters[name]
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f'Categorical flow {name} must be a real number')
            if not math.isfinite(value):
                raise ValueError(f'Categorical flow {name} must be finite')
            parameters[name] = float(value)
        for name in ('time_step', 'mobility', 'schedule_exponent'):
            if parameters[name] <= 0:
                raise ValueError(f'Categorical flow {name} must be positive')
        if parameters['route_strength'] < 0 or parameters['gamma'] < 0:
            raise ValueError('Categorical flow route_strength and gamma must be nonnegative')
        if parameters['kappa_final'] < parameters['kappa_initial']:
            raise ValueError('Categorical flow kappa_final must be at least kappa_initial')
        if not 0 <= parameters['sparse_threshold'] <= 1:
            raise ValueError('Categorical flow sparse_threshold must be between zero and one')
        for name, choices in (('dtype', ('float32', 'float64')),
                              ('integrator', ('euler', 'heun')),
                              ('matrix_format', ('auto', 'dense', 'sparse'))):
            if parameters[name] not in choices:
                raise ValueError(f'Categorical flow {name} must be one of {choices}')
        for name in ('cuda_graph', 'deduplicate_candidates'):
            if not isinstance(parameters[name], bool):
                raise TypeError(f'Categorical flow {name} must be a boolean')
        return parameters

    def estimatedWorkingMemoryBytes(self, problem, solverParameters=None):
        p = self._getParameters(solverParameters)
        n = problem.variableCount
        if n > p['max_variables']:
            raise ValueError('Categorical flow problem exceeds max_variables')
        if p['matrix_format'] == 'dense' and n > p['max_dense_variables']:
            raise ValueError('Categorical flow dense matrix exceeds max_dense_variables')
        width = min(p['run_batch_size'] or p['runs'], p['runs'])
        candidates = max(width, p['candidate_batch_size'])
        size = 4 if p['dtype'] == 'float32' else 8
        dense = n * n * size if p['matrix_format'] != 'sparse' and n <= p['max_dense_variables'] else 0
        return int(12 * problem.numericMemoryBytes + dense + n * width * size * 24
                   + n * candidates * 32
                   + min(problem.interactionCount, p['energy_chunk_size']) * candidates * 32
                   + (p['steps'] + 1) * size)

    def _solveBatch(self, problems, parameters):
        # Avoid an O((sum N)^2) allocation for explicitly dense scenario batches.
        if parameters['matrix_format'] == 'dense' and len(problems) > 1:
            solve_batch = super()._solveBatch
            return [result for problem in problems
                    for result in solve_batch([problem], parameters)]
        return super()._solveBatch(problems, parameters)

    def solveResidentPlanned(self, problems, plan, solverParameters=None):
        """Normalize the authoritative host snapshot, including duplicate edges."""
        if len(self.devices) > 1:
            raise ValueError('Resident categorical flow requires one solver device')
        if plan.shards != ((0, len(problems)),) and problems:
            raise ValueError('Resident resource plan must cover the ordered batch')
        return self.solvePlanned([problem.source for problem in problems], plan, solverParameters)

