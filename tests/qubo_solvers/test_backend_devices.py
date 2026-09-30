"""Every GPU-capable canonical entry must execute its search on the requested device.

The dispatch observer is installed inside each algorithm's search, after host
topology/preparation. Thus GPU coverage cannot pass merely by copying an output
to CUDA. CPU parametrizations exercise the same instrumentation on local hosts.
"""

import functools

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from qubo_solvers import (
    CATEGORICAL_SOLVERS,
    GPU_SOLVERS,
    NATIVE_SOLVERS,
    create_bqm_solver,
    library_solver_class,
)
from qubo_solvers.backends.problem import QUBOProblem
from qubo_solvers.registry import TREE_SOLVERS


PHYSICS_SOLVERS = (
    "lib_altermagnet", "lib_dynamical_geometry", "lib_geometric",
    "lib_phonon_exchange", "lib_supersymmetric",
)
CUDA = pytest.param(
    "cuda:0", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA hardware/runtime is unavailable",
    ),
)


def _problem(name):
    return QUBOProblem(
        [-1., 2., -3., .5],
        [0, 1, 0, 2, 2, 3], [1, 0, 1, 2, 3, 2],
        [.75, -.25, .5, 1.125, -2., .5], offset=7.5, seedOffset=19,
        oneHotGroups=((0, 1), (2, 3)) if name in CATEGORICAL_SOLVERS else (),
    )


def _parameters(name, precision):
    parameters = dict(steps=2, runs=2, run_batch_size=1, dtype=precision, seed=11)
    if name in TREE_SOLVERS:
        parameters.pop("steps")
        parameters.pop("run_batch_size")
        parameters["max_treewidth"] = 3
    elif name in PHYSICS_SOLVERS:
        parameters.update(candidate_interval=1, conditional_rounding=False,
                          local_search_steps=0, matrix_format="dense")
    elif name == "lib_categorical_trf":
        # Observe eager search kernels rather than graph-capture setup/replay.
        parameters.update(cuda_graph=False, candidate_interval=1)
    return parameters


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)


class _SearchArithmetic(TorchDispatchMode):
    # Copies, allocations and comparisons are deliberately not evidence of
    # numerical execution. In-place variants share the same stripped name.
    operations = {
        "mm", "bmm", "addmm", "_sparse_addmm", "mul", "add", "sub", "div",
        "addcmul", "sin", "cos", "tanh", "sigmoid", "sum", "min", "max",
        "amin", "amax", "logsumexp", "bernoulli", "exp", "sqrt",
    }

    def __init__(self, observed):
        super().__init__()
        self.observed = observed

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        result = function(*args, **(kwargs or {}))
        name = function._schema.name.split("::")[-1].rstrip("_")
        if name in self.operations:
            self.observed.extend(
                (name, tensor.device) for tensor in _tensors(result)
                if tensor.is_floating_point()
            )
        return result


def _observe_search(monkeypatch, solver, name, expected_device, expected_dtype):
    observed, calls = [], []

    def check(*tensors):
        for tensor in tensors:
            assert tensor.device == expected_device
            assert tensor.dtype == expected_dtype

    def instrument(owner, attribute, check_arguments, *, static=False):
        original = getattr(owner, attribute)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            check_arguments(args)
            calls.append(attribute)
            with _SearchArithmetic(observed):
                return original(*args, **kwargs)

        monkeypatch.setattr(owner, attribute, staticmethod(wrapped) if static else wrapped)

    if name in NATIVE_SOLVERS:
        instrument(library_solver_class(name), "_search",
                   lambda args: check(args[1].q, args[1].x))
    elif name in PHYSICS_SOLVERS:
        instrument(type(solver), "_run", lambda args: check(args[1].h, args[1].matrix))
    elif name == "lib_simulated_bifurcation":
        instrument(type(solver), "_runTrajectories", lambda args: check(args[2], args[3]))
    elif name == "lib_categorical":
        instrument(type(solver), "_runCategorical", lambda args: check(args[2], args[3]),
                   static=True)
    elif name == "lib_categorical_trf":
        from qubo_solvers.backends.categorical_trf import _CategoricalFlowWorkspace

        instrument(_CategoricalFlowWorkspace, "advance",
                   lambda args: check(args[0].theta, args[0].matrix))
    elif name in TREE_SOLVERS:
        from qubo_solvers.backends import tree_decomposition

        def check_tree(args):
            assert torch.device(args[4]) == expected_device
            assert getattr(torch, args[3]["dtype"]) == expected_dtype

        instrument(tree_decomposition, "_eliminate", check_tree)
    else:
        pytest.fail(f"No search instrumentation for canonical solver {name}")
    return observed, calls


@pytest.mark.parametrize("solver_name", GPU_SOLVERS)
@pytest.mark.parametrize("device_name", ["cpu", CUDA])
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_canonical_compact_search_device_and_source_score(
    monkeypatch, solver_name, device_name, precision,
):
    problem = _problem(solver_name)
    arrays = (problem.linear, problem.quadraticHeads, problem.quadraticTails,
              problem.quadraticBiases, problem.groupOffsets)
    snapshots = tuple(array.copy() for array in arrays)
    source_metadata = (problem.offset, problem.seedOffset, problem.oneHotGroups)
    solver = create_bqm_solver(solver_name, {"device": device_name})
    expected_device = torch.device(device_name)
    assert torch.device(solver.device) == expected_device
    observed, calls = _observe_search(
        monkeypatch, solver, solver_name, expected_device, getattr(torch, precision),
    )

    result = solver.solve(problem, _parameters(solver_name, precision))
    if expected_device.type == "cuda":
        torch.cuda.synchronize(expected_device)

    assert calls, "The instrumented algorithm search was never invoked"
    assert observed, "No floating-point search arithmetic was observed"
    assert {device for _, device in observed} == {expected_device}
    bits = tuple(result.sample)
    assert len(bits) == 4 and all(bit in (0, 1) for bit in bits)
    x0, x1, x2, x3 = bits
    # Independent pair-once objective, including all duplicate, reversed and
    # diagonal input terms. Deliberately score in original Python/FP64 values.
    expected_energy = (7.5 - x0 + 2.*x1 - 1.875*x2 + .5*x3
                       + x0*x1 - 1.5*x2*x3)
    assert result.energy == expected_energy == problem.energy(bits)
    if solver_name in CATEGORICAL_SOLVERS:
        assert x0 + x1 == x2 + x3 == 1
    for original, snapshot in zip(arrays, snapshots):
        np.testing.assert_array_equal(original, snapshot)
        assert not original.flags.writeable
    assert (problem.offset, problem.seedOffset, problem.oneHotGroups) == source_metadata


@pytest.mark.parametrize("solver_name", GPU_SOLVERS)
def test_explicit_unavailable_cuda_never_falls_back(monkeypatch, solver_name):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # Some backends resolve lazily at .device; others validate construction.
    with pytest.raises((RuntimeError, ValueError), match="(?i)cuda.*(available|support)"):
        solver = create_bqm_solver(solver_name, {"device": "cuda:0"})
        _ = solver.device
        solver.solve(_problem(solver_name), _parameters(solver_name, "float32"))
