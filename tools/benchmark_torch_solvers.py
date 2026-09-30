"""Profile synchronized SBM/SVL solves on deterministic sparse QUBOs.

Run from the repository root with PYTHONPATH=src. Use identical arguments on
both revisions; JSON includes source hashes, inputs, energies and sample hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy
import torch

from margin_calculator.optimization.optimization_problem.qubo_problem import QUBOProblem
from qubo_solvers import create_bqm_solver
from qubo_solvers.backends.candidate_selection import CandidateSelection
from qubo_solvers.backends.torch_candidates import TorchCandidateAccumulator
from qubo_solvers.backends.torch_execution import TorchExecution
from benchmark_biqmac import librarySourceHashes


def makeProblems(variables: int, count: int, degree: int, oneHot: bool) -> list[QUBOProblem]:
    """Build sparse fixtures without allocating a variables-squared array."""
    rng = numpy.random.default_rng(13)
    problems = []
    for _ in range(count):
        heads = numpy.repeat(numpy.arange(variables, dtype=numpy.uint32), degree)
        tails = rng.integers(0, variables, len(heads), dtype=numpy.uint32)
        biases = rng.normal(0, .1, len(heads))
        linear = rng.normal(0, .1, variables)
        groups = tuple(tuple(range(start, min(start + 4, variables)))
                       for start in range(0, variables, 4)) if oneHot else ()
        if groups:
            pairs = [(head, tail) for group in groups
                     for index, head in enumerate(group) for tail in group[index + 1:]]
            heads = numpy.concatenate((heads, numpy.array([h for h, _ in pairs], dtype=numpy.uint32)))
            tails = numpy.concatenate((tails, numpy.array([t for _, t in pairs], dtype=numpy.uint32)))
            biases = numpy.concatenate((biases, numpy.full(len(pairs), 12.)))
            linear -= 6.
        problems.append(QUBOProblem(linear, heads, tails, biases,
                                   offset=6. * len(groups), oneHotGroups=groups))
    return problems


def measure(solver, problems, parameters, warmups, repeats, trace):
    device = torch.device(solver.device)
    devices = tuple(torch.device(item) for item in getattr(solver, 'devices', (solver.device,)))
    gpu_devices = tuple(item for item in devices if item.type == 'cuda')
    gpu = bool(gpu_devices)

    def synchronize():
        for item in gpu_devices:
            torch.cuda.synchronize(item)

    synchronize()
    start = perf_counter()
    solver.solveMany(problems, parameters)
    synchronize()
    cold = perf_counter() - start
    for _ in range(warmups):
        solver.solveMany(problems, parameters)
    synchronize()
    for item in gpu_devices:
        torch.cuda.reset_peak_memory_stats(item)
    timings = []
    results = None
    for _ in range(repeats):
        synchronize()
        start = perf_counter()
        results = solver.solveMany(problems, parameters)
        synchronize()
        timings.append(perf_counter() - start)
    peak = {str(item): torch.cuda.max_memory_allocated(item) for item in gpu_devices} or None
    signatures = []
    for problem, result in zip(problems, results):
        sample = numpy.asarray(result.sample)
        assert sample.shape == (problem.variableCount,)
        assert numpy.isin(sample, [0, 1]).all()
        assert all(sample[list(group)].sum() == 1 for group in problem.iterOneHotGroups())
        numpy.testing.assert_allclose(result.energy, problem.energy(sample), rtol=0, atol=1e-12)
        signatures.append({"energy": result.energy,
                           "sample_sha256": hashlib.sha256(sample.astype(numpy.uint8).tobytes()).hexdigest()})
    profile_summary = None
    if trace is not None:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if gpu:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, profile_memory=True,
                                    record_shapes=True) as profile:
            solver.solveMany(problems, parameters)
            synchronize()
        trace.parent.mkdir(parents=True, exist_ok=True)
        profile.export_chrome_trace(str(trace))
        profile_summary = profile.key_averages().table(
            sort_by="self_cuda_time_total" if gpu else "self_cpu_time_total", row_limit=20)
    return {"cold_seconds": cold, "seconds": timings, "median_seconds": median(timings),
            "peak_cuda_allocated_bytes": peak, "results": signatures,
            "profile": profile_summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--variables", type=int, nargs="+", default=[64, 1024, 8192])
    parser.add_argument("--problems", type=int, default=1)
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--one-hot", action="store_true")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--runs", type=int, default=16)
    parser.add_argument("--run-batch-size", type=int)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--integrator", choices=["euler", "heun"], default="heun")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--trace-directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(*args.variables, args.problems, args.degree, args.steps, args.runs,
           args.repeats, args.threads) <= 0 or args.warmups < 0:
        parser.error("sizes, repeats and threads must be positive; warmups must be nonnegative")
    if args.run_batch_size is not None and args.run_batch_size <= 0:
        parser.error("run-batch-size must be positive")
    torch.set_num_threads(args.threads)
    resolved = create_bqm_solver('lib_simulated_bifurcation', {'device': args.device}).device
    device = torch.device(resolved)
    metadata = {"python": platform.python_version(), "platform": platform.platform(),
                "torch": torch.__version__, "numpy": numpy.__version__,
                "cuda": torch.version.cuda, "hip": torch.version.hip,
                "device": resolved, "threads": torch.get_num_threads(),
                "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
                "arguments": {key: str(value) if isinstance(value, Path) else value
                              for key, value in vars(args).items()}}
    metadata["shared_source_sha256"] = {
        cls.__name__: hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest()
        for cls in (TorchExecution, TorchCandidateAccumulator, CandidateSelection)
    }
    metadata["library_source_sha256"] = librarySourceHashes()
    measurements = []
    for variables in args.variables:
        problems = makeProblems(variables, args.problems, args.degree, args.one_hot)
        for name in ('lib_simulated_bifurcation', 'lib_spin_vector_langevin'):
            solver = create_bqm_solver(name, {'device': resolved})
            parameters = {"steps": args.steps, "runs": args.runs, "seed": 13,
                          "dtype": args.dtype, "run_batch_size": args.run_batch_size}
            if name == 'lib_spin_vector_langevin':
                parameters["integrator"] = args.integrator
            trace = args.trace_directory / f"{name}-{variables}.json" if args.trace_directory else None
            result = measure(solver, problems, parameters, args.warmups, args.repeats, trace)
            result.update({"solver": name, "variables": variables,
                           "interactions": [problem.interactionCount for problem in problems],
                           "input_seeds": [int(problem.seedOffset) for problem in problems],
                           "parameters": solver._getParameters(parameters),
                           "source_sha256": hashlib.sha256(Path(inspect.getfile(type(solver))).read_bytes()).hexdigest()})
            measurements.append(result)
            print(f"{name} variables={variables}: {result['median_seconds']:.6f}s", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"environment": metadata, "measurements": measurements}, indent=2) + "\n")


if __name__ == "__main__":
    main()
