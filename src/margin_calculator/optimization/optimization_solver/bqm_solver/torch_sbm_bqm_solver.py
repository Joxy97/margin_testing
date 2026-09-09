"""Simulated-bifurcation dynamics on the shared Torch execution runtime."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy

from .bqm_solver_factory import BQMSolverFactory
from .torch_execution import TorchExecution, _MAX_TORCH_SEED, _RUN_SEED_STRIDE


class TorchSBMBQMSolver(TorchExecution):
    """Batched simulated bifurcation with device-independent seeded trajectories."""

    _defaults = {
        "steps": 10_000,
        "runs": 16,
        "dt": 1.0,
        "a0": 1.0,
        "c0": 0.0,
        "gamma": 0.0,
        "initial_scale": 0.05,
        "seed": 1,
        "dtype": "float32",
        "run_batch_size": None,
        "energy_chunk_size": 1_000_000,
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
        positions = torch.empty(shape, dtype=torchDtype, device=device)
        momenta = torch.empty_like(positions)
        for problem_index in range(len(variableOffsets) - 1):
            start = int(variableOffsets[problem_index])
            stop = int(variableOffsets[problem_index + 1])
            for local_run in range(width):
                run = runStart + local_run
                seed = (
                    parameters["seed"]
                    + int(problemSeedOffsets[problem_index])
                    + _RUN_SEED_STRIDE * run
                ) % _MAX_TORCH_SEED
                generator = torch.Generator(device=device)
                generator.manual_seed(seed)
                positions[start:stop, local_run].uniform_(
                    -parameters["initial_scale"],
                    parameters["initial_scale"],
                    generator=generator,
                )
                momenta[start:stop, local_run].uniform_(
                    -parameters["initial_scale"],
                    parameters["initial_scale"],
                    generator=generator,
                )

        scratch = torch.empty_like(positions)
        force = torch.empty_like(positions)
        wall = torch.empty(shape, dtype=torch.bool, device=device)
        old_momenta = (
            torch.empty_like(momenta) if parameters["gamma"] else None
        )
        for step in range(parameters["steps"]):
            torch.sign(positions, out=scratch)
            scratch.masked_fill_(scratch == 0, 1.0)
            if old_momenta is not None:
                old_momenta.copy_(momenta)
            # beta=0 ignores the previous contents, including NaNs. Reuse the
            # output instead of allocating and zeroing a sparse.mm result.
            torch.addmm(force, matrix, scratch, beta=0, out=force)
            force.add_(field).mul_(c0Rows)
            pressure = parameters["a0"] * step / parameters["steps"]
            force.add_(positions, alpha=pressure - parameters["a0"])
            momenta.add_(force, alpha=parameters["dt"])
            positions.add_(
                momenta,
                alpha=parameters["a0"] * parameters["dt"],
            )
            torch.abs(positions, out=scratch)
            torch.gt(scratch, 1.0, out=wall)
            positions.clamp_(-1.0, 1.0)
            momenta.masked_fill_(wall, 0.0)
            if old_momenta is not None:
                momenta.add_(
                    old_momenta,
                    alpha=parameters["gamma"] * parameters["dt"],
                )

        torch.sign(positions, out=scratch)
        scratch.masked_fill_(scratch == 0, 1.0)
        return ((scratch + 1.0) * 0.5).to(torch.uint8)

    @classmethod
    def _getParameters(
        cls,
        solverParameters: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        supplied = dict(solverParameters or {})
        unknown = supplied.keys() - cls._defaults.keys()
        if unknown:
            raise ValueError(
                f"Unknown Torch SBM parameters: {sorted(unknown)}"
            )
        values = cls._defaults | supplied
        normalized = {
            "steps": int(values["steps"]),
            "runs": int(values["runs"]),
            "dt": float(values["dt"]),
            "a0": float(values["a0"]),
            "c0": float(values["c0"]),
            "gamma": float(values["gamma"]),
            "initial_scale": float(values["initial_scale"]),
            "seed": int(values["seed"]),
            "dtype": str(values["dtype"]),
            "run_batch_size": (
                None
                if values["run_batch_size"] is None
                else int(values["run_batch_size"])
            ),
            "energy_chunk_size": int(values["energy_chunk_size"]),
        }
        if normalized["steps"] <= 0 or normalized["runs"] <= 0:
            raise ValueError("Torch SBM steps and runs must be positive")
        if normalized["dt"] <= 0.0 or normalized["a0"] <= 0.0:
            raise ValueError("Torch SBM dt and a0 must be positive")
        if normalized["c0"] < 0.0 or normalized["gamma"] < 0.0:
            raise ValueError("Torch SBM c0 and gamma must be nonnegative")
        if normalized["initial_scale"] < 0.0:
            raise ValueError("Torch SBM initial_scale must be nonnegative")
        if not 0 <= normalized["seed"] < _MAX_TORCH_SEED:
            raise ValueError("Torch SBM seed must be in [0, 2**63 - 1)")
        if normalized["dtype"] not in {"float32", "float64"}:
            raise ValueError("Torch SBM dtype must be float32 or float64")
        if (
            normalized["run_batch_size"] is not None
            and normalized["run_batch_size"] <= 0
        ):
            raise ValueError("Torch SBM run_batch_size must be positive")
        if normalized["energy_chunk_size"] <= 0:
            raise ValueError("Torch SBM energy_chunk_size must be positive")
        return normalized


BQMSolverFactory.registerSolver("torch_sbm", TorchSBMBQMSolver)
