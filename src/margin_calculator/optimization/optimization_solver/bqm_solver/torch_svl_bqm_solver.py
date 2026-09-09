"""Spin-vector Langevin QUBO solver implemented with PyTorch."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy

from margin_calculator.optimization.optimization_result import (
    BQMOptimizationResult,
)

from ...optimization_problem.qubo_problem import QUBOProblem
from .bqm_solver_factory import BQMSolverFactory
from .torch_candidates import TorchCandidateAccumulator
from .torch_execution import (
    _MAX_TORCH_SEED,
    _RUN_SEED_STRIDE,
    TorchExecution,
)


class TorchSVLBQMSolver(TorchExecution):
    """Solve QUBOs with batched spin-vector Langevin trajectories.

    Each binary variable is represented by the sign of ``sin(theta)``. The
    continuous angles and momenta follow underdamped Langevin dynamics while
    transverse-field and problem scales are annealed linearly. Independent
    replicas are batched as tensor columns.
    """

    _defaults = {
        "steps": 2_000,
        "runs": 32,
        "dt": 0.01,
        "mass": 1.0,
        "damping": 0.1,
        "temperature": 0.01,
        "transverse_field_initial": 1.0,
        "transverse_field_final": 0.0,
        "problem_scale_initial": 0.0,
        "problem_scale_final": 1.0,
        "seed": 1,
        "dtype": "float32",
        "integrator": "euler_maruyama",
        "run_batch_size": None,
        "energy_chunk_size": 1_000_000,
        "noise_chunk_size": 16,
    }

    @staticmethod
    def _runTrajectories(
        torch: Any,
        matrix: Any,
        field: Any,
        c0Rows: Any,
        variableOffsets: numpy.ndarray,
        problemSeedOffsets: numpy.ndarray,
        width: int,
        runStart: int,
        parameters: Mapping[str, Any],
        torchDtype: Any,
        device: Any,
    ) -> Any:
        shape = (int(variableOffsets[-1]), width)
        angles = torch.empty(shape, dtype=torchDtype, device=device)
        generators = []
        for problem_index in range(len(variableOffsets) - 1):
            start = int(variableOffsets[problem_index])
            stop = int(variableOffsets[problem_index + 1])
            problem_generators = []
            for local_run in range(width):
                run = runStart + local_run
                seed = (
                    parameters["seed"]
                    + int(problemSeedOffsets[problem_index])
                    + _RUN_SEED_STRIDE * run
                ) % _MAX_TORCH_SEED
                generator = torch.Generator(device=device)
                generator.manual_seed(seed)
                # Start close to the transverse-field ground state.  The
                # small seeded perturbation also breaks symmetry when T=0.
                angles[start:stop, local_run].normal_(
                    mean=0.0, std=1.0e-3, generator=generator
                )
                problem_generators.append(generator)
            generators.append(problem_generators)

        velocities = torch.zeros_like(angles)
        spins = torch.empty_like(angles)
        local_field = torch.empty_like(angles)
        longitudinal = torch.empty_like(angles)
        transverse_force = torch.empty_like(angles)

        def force(values: Any, transverse: float, problem_scale: float) -> Any:
            # Only scratch storage is shared between force evaluations. The
            # caller consumes it into a separate acceleration before reuse.
            torch.sin(values, out=spins)
            torch.addmm(local_field, matrix, spins, beta=0, out=local_field)
            local_field.add_(field)
            torch.cos(values, out=longitudinal)
            longitudinal.mul_(problem_scale).mul_(local_field)
            torch.mul(spins, -transverse, out=transverse_force)
            return transverse_force.add_(longitudinal)

        noise_scale = math.sqrt(2.0 * parameters["damping"] * parameters["temperature"]
                                * parameters["dt"]) / parameters["mass"]
        chunk_size = min(parameters["noise_chunk_size"], parameters["steps"])
        if not noise_scale:
            chunk_size = 1
        noise_buffer = torch.zeros((chunk_size, *shape), dtype=torchDtype, device=device)
        for step in range(parameters["steps"]):
            fraction = step / max(parameters["steps"] - 1, 1)
            transverse = (
                parameters["transverse_field_initial"]
                + fraction * (
                    parameters["transverse_field_final"]
                    - parameters["transverse_field_initial"]
                )
            )
            problem_scale = (
                parameters["problem_scale_initial"]
                + fraction * (
                    parameters["problem_scale_final"]
                    - parameters["problem_scale_initial"]
                )
            )
            if noise_scale and step % chunk_size == 0:
                # One contiguous RNG draw per trajectory for several steps.
                # Stream identity is independent of problem/run batch layout.
                for problem_index, problem_generators in enumerate(generators):
                    start = int(variableOffsets[problem_index])
                    stop = int(variableOffsets[problem_index + 1])
                    for local_run, generator in enumerate(problem_generators):
                        block = torch.randn((chunk_size, stop - start), generator=generator,
                                            dtype=torchDtype, device=device)
                        noise_buffer[:, start:stop, local_run].copy_(block)
                noise_buffer.mul_(noise_scale)
            noise = noise_buffer[step % chunk_size]
            first_acceleration = (
                force(angles, transverse, problem_scale)
                - parameters["damping"] * velocities
            ) / parameters["mass"]
            if parameters["integrator"] == "euler_maruyama":
                angles.add_(velocities, alpha=parameters["dt"])
                velocities.add_(
                    first_acceleration, alpha=parameters["dt"]
                ).add_(noise)
            else:
                predicted_angles = angles + parameters["dt"] * velocities
                predicted_velocities = (
                    velocities
                    + parameters["dt"] * first_acceleration
                    + noise
                )
                next_fraction = min(
                    (step + 1) / max(parameters["steps"] - 1, 1), 1.0
                )
                next_transverse = (
                    parameters["transverse_field_initial"]
                    + next_fraction
                    * (
                        parameters["transverse_field_final"]
                        - parameters["transverse_field_initial"]
                    )
                )
                next_problem_scale = (
                    parameters["problem_scale_initial"]
                    + next_fraction
                    * (
                        parameters["problem_scale_final"]
                        - parameters["problem_scale_initial"]
                    )
                )
                second_acceleration = (
                    force(
                        predicted_angles,
                        next_transverse,
                        next_problem_scale,
                    )
                    - parameters["damping"] * predicted_velocities
                ) / parameters["mass"]
                angles.add_(
                    velocities + predicted_velocities,
                    alpha=0.5 * parameters["dt"],
                )
                velocities.add_(
                    first_acceleration + second_acceleration,
                    alpha=0.5 * parameters["dt"],
                ).add_(noise)
            angles.remainder_(2.0 * math.pi)
        return (torch.sin(angles) >= 0.0).to(torch.uint8)

    @classmethod
    def _getParameters(
        cls, solverParameters: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        supplied = dict(solverParameters or {})
        unknown = supplied.keys() - cls._defaults.keys()
        if unknown:
            raise ValueError(f"Unknown Torch SVL parameters: {sorted(unknown)}")
        values = cls._defaults | supplied
        parameters = {
            "steps": int(values["steps"]),
            "runs": int(values["runs"]),
            "dt": float(values["dt"]),
            "mass": float(values["mass"]),
            "damping": float(values["damping"]),
            "temperature": float(values["temperature"]),
            "transverse_field_initial": float(values["transverse_field_initial"]),
            "transverse_field_final": float(values["transverse_field_final"]),
            "problem_scale_initial": float(values["problem_scale_initial"]),
            "problem_scale_final": float(values["problem_scale_final"]),
            "seed": int(values["seed"]),
            "dtype": str(values["dtype"]),
            "integrator": str(values["integrator"]),
            "run_batch_size": (
                None
                if values["run_batch_size"] is None
                else int(values["run_batch_size"])
            ),
            "energy_chunk_size": int(values["energy_chunk_size"]),
            "noise_chunk_size": int(values["noise_chunk_size"]),
        }
        if parameters["steps"] <= 0 or parameters["runs"] <= 0:
            raise ValueError("Torch SVL steps and runs must be positive")
        if parameters["dt"] <= 0.0 or parameters["mass"] <= 0.0:
            raise ValueError("Torch SVL dt and mass must be positive")
        for name in (
            "damping",
            "temperature",
            "transverse_field_initial",
            "transverse_field_final",
            "problem_scale_initial",
            "problem_scale_final",
        ):
            if parameters[name] < 0.0:
                raise ValueError(f"Torch SVL {name} must be nonnegative")
        if not 0 <= parameters["seed"] < _MAX_TORCH_SEED:
            raise ValueError("Torch SVL seed must be in [0, 2**63 - 1)")
        if parameters["dtype"] not in {"float32", "float64"}:
            raise ValueError("Torch SVL dtype must be float32 or float64")
        if parameters["integrator"] not in {
            "euler_maruyama",
            "weak_order_2",
        }:
            raise ValueError(
                "Torch SVL integrator must be euler_maruyama or weak_order_2"
            )
        if (
            parameters["run_batch_size"] is not None
            and parameters["run_batch_size"] <= 0
        ):
            raise ValueError("Torch SVL run_batch_size must be positive")
        if parameters["energy_chunk_size"] <= 0:
            raise ValueError("Torch SVL energy_chunk_size must be positive")
        if parameters["noise_chunk_size"] <= 0:
            raise ValueError("Torch SVL noise_chunk_size must be positive")
        for name, value in parameters.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"Torch SVL {name} must be finite")
        return parameters


BQMSolverFactory.registerSolver("torch_svl", TorchSVLBQMSolver)
