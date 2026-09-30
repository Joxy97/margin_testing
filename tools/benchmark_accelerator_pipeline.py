"""Compare PCA backends and library SVL restart batching on local inputs.

Run with PYTHONPATH=src; --device cuda:0 enables synchronized GPU timing.
"""

import argparse
import json
from statistics import median
from time import perf_counter

import numpy
import torch

from risk_state_generator import NumpyPCABackend, TorchPCABackend
from qubo_solvers import create_bqm_solver
from margin_calculator.optimization.optimization_problem.qubo_problem import QUBOProblem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--assets", type=int, default=512)
    parser.add_argument("--observations", type=int, default=60)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--runs", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(args.assets, args.observations, args.steps, args.runs, args.repeats) <= 0:
        parser.error("sizes and repeats must be positive")
    rng = numpy.random.default_rng(13)
    values = rng.normal(size=(args.observations, args.assets))
    weights = .94 ** numpy.arange(args.observations - 1, -1, -1)
    weights /= weights.sum()
    numpy_backend = NumpyPCABackend()
    torch_backend = TorchPCABackend(args.device)
    components = min(3, args.assets, args.observations - 1)
    expected = numpy_backend.fit(values, weights, components)
    actual = torch_backend.fit(values, weights, components)
    for name in vars(expected):
        numpy.testing.assert_allclose(getattr(actual, name), getattr(expected, name), atol=1e-10)

    def timed(function):
        function()
        timings = []
        for _ in range(args.repeats):
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(args.device)
            start = perf_counter()
            function()
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(args.device)
            timings.append(perf_counter() - start)
        return median(timings)

    problems = [QUBOProblem(
        rng.normal(size=64), numpy.arange(63, dtype=numpy.uint32),
        numpy.arange(1, 64, dtype=numpy.uint32), rng.normal(size=63)
    ) for _ in range(4)]
    solver = create_bqm_solver('lib_spin_vector_langevin', {'device': args.device})
    parameters = {"steps": args.steps, "runs": args.runs, "seed": 13}
    results = {
        "device": args.device, "torch_version": torch.__version__,
        "pca_numpy_seconds": timed(lambda: numpy_backend.fit(values, weights, components)),
        "pca_torch_seconds": timed(lambda: torch_backend.fit(values, weights, components)),
        "svl_batch_1_seconds": timed(lambda: solver.solveMany(problems, parameters | {"run_batch_size": 1})),
        "svl_batch_all_seconds": timed(lambda: solver.solveMany(problems, parameters | {"run_batch_size": args.runs})),
    }
    results["restart_batch_speedup"] = results["svl_batch_1_seconds"] / results["svl_batch_all_seconds"]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
