"""Compare sparse solver trajectories with dense integration equations."""

from importlib.util import find_spec
import math
import unittest

import numpy

from margin_calculator.optimization.optimization_solver.bqm_solver import (
    TorchSBMBQMSolver,
    TorchSVLBQMSolver,
)
from margin_calculator.optimization.optimization_solver.bqm_solver.torch_execution import (
    _MAX_TORCH_SEED,
    _RUN_SEED_STRIDE,
)


class _CaptureTorch:
    """Observe the continuous state at the final spin conversion."""

    def __init__(self, torch):
        self.torch = torch
        self.state = None

    def __getattr__(self, name):
        return getattr(self.torch, name)

    def sign(self, values, **kwargs):
        self.state = values.clone()
        return self.torch.sign(values, **kwargs)

    def sin(self, values, **kwargs):
        if "out" not in kwargs:
            self.state = values.clone()
        return self.torch.sin(values, **kwargs)


def reference(torch, solver, matrix, field, scales, offsets, seeds, width, runStart, p, device):
    """Use the same random streams, then integrate independently in NumPy."""
    svl = solver is TorchSVLBQMSolver
    positions = torch.empty((int(offsets[-1]), width), dtype=torch.float64, device=device)
    momenta = torch.zeros_like(positions)
    generators = []
    for index, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        for run in range(width):
            seed = (p["seed"] + int(seeds[index]) + _RUN_SEED_STRIDE * (runStart + run)) % _MAX_TORCH_SEED
            generator = torch.Generator(device=device).manual_seed(seed)
            if svl:
                positions[start:stop, run].normal_(0, 1e-3, generator=generator)
            else:
                positions[start:stop, run].uniform_(-p["initial_scale"], p["initial_scale"], generator=generator)
                momenta[start:stop, run].uniform_(-p["initial_scale"], p["initial_scale"], generator=generator)
            generators.append((start, stop, run, generator))
    x, v = positions.cpu().numpy().copy(), momenta.cpu().numpy().copy()
    if not svl:
        for step in range(p["steps"]):
            old_v = v.copy()
            force = (matrix @ numpy.where(x >= 0, 1., -1.) + field) * scales
            force += (p["a0"] * step / p["steps"] - p["a0"]) * x
            v += p["dt"] * force
            x += p["a0"] * p["dt"] * v
            v[numpy.abs(x) > 1.] = 0.
            x = numpy.clip(x, -1., 1.)
            v += p["gamma"] * p["dt"] * old_v
        return x, (x >= 0).astype(numpy.uint8)

    noise_scale = math.sqrt(2 * p["damping"] * p["temperature"] * p["dt"]) / p["mass"]
    chunk = min(p["noise_chunk_size"], p["steps"]) if noise_scale else 1
    noise = numpy.zeros((chunk, *x.shape))

    def acceleration(values, velocities, step):
        fraction = min(step / max(p["steps"] - 1, 1), 1.)
        transverse = p["transverse_field_initial"] + fraction * (p["transverse_field_final"] - p["transverse_field_initial"])
        scale = p["problem_scale_initial"] + fraction * (p["problem_scale_final"] - p["problem_scale_initial"])
        force = -transverse * numpy.sin(values) + scale * numpy.cos(values) * (matrix @ numpy.sin(values) + field)
        return (force - p["damping"] * velocities) / p["mass"]

    for step in range(p["steps"]):
        if noise_scale and step % chunk == 0:
            for start, stop, run, generator in generators:
                noise[:, start:stop, run] = torch.randn(
                    (chunk, stop - start), generator=generator, dtype=torch.float64, device=device
                ).cpu().numpy()
            noise *= noise_scale
        kick = noise[step % chunk]
        first = acceleration(x, v, step)
        if p["integrator"] == "euler_maruyama":
            x += p["dt"] * v
            v += p["dt"] * first + kick
        else:
            predicted_x = x + p["dt"] * v
            predicted_v = v + p["dt"] * first + kick
            second = acceleration(predicted_x, predicted_v, step + 1)
            x += .5 * p["dt"] * (v + predicted_v)
            v += .5 * p["dt"] * (first + second) + kick
        x %= 2 * math.pi
    return x, (numpy.sin(x) >= 0).astype(numpy.uint8)


@unittest.skipUnless(find_spec("torch"), "requires torch")
class TorchDynamicsTest(unittest.TestCase):
    def check_device(self, device):
        import torch

        dense = numpy.array([[0., .3, -.2, 0., 0.], [.3, 0., .5, 0., 0.],
                             [-.2, .5, 0., 0., 0.], [0., 0., 0., 0., -.7],
                             [0., 0., 0., -.7, 0.]])
        field = numpy.array([-.7, .2, .4, -.3, .6])[:, None]
        scales = numpy.array([.3, .3, .3, .5, .5])[:, None]
        offsets, seeds = numpy.array([0, 3, 5]), numpy.array([17, 31], dtype=numpy.uint64)
        cases = [(TorchSBMBQMSolver, {"gamma": gamma, "dt": .7, "initial_scale": .8}) for gamma in (0., .2)]
        cases += [(TorchSVLBQMSolver, {"integrator": integrator, "temperature": temperature,
                                       "noise_chunk_size": 8, "dt": .07, "mass": 1.3})
                  for integrator in ("euler_maruyama", "weak_order_2") for temperature in (0., .03)]
        for solver, settings in cases:
            for empty in (False, True):
                for steps in (1, 19):
                    with self.subTest(device=device, solver=solver.__name__, settings=settings,
                                      empty=empty, steps=steps), torch.inference_mode():
                        matrix = numpy.zeros_like(dense) if empty else dense
                        p = solver._getParameters(settings | {"steps": steps, "runs": 3, "dtype": "float64"})
                        expected_state, expected_samples = reference(
                            torch, solver, matrix, field, scales, offsets, seeds, 3, 2, p, device)
                        capture = _CaptureTorch(torch)
                        samples = solver._runTrajectories(
                            capture, torch.tensor(matrix, device=device).to_sparse_csr(),
                            torch.tensor(field, device=device), torch.tensor(scales, device=device),
                            offsets, seeds, 3, 2, p, torch.float64, torch.device(device))
                        numpy.testing.assert_allclose(capture.state.cpu().numpy(), expected_state,
                                                      rtol=1e-11, atol=1e-12)
                        numpy.testing.assert_array_equal(samples.cpu().numpy(), expected_samples)

    def test_cpu_matches_dense_equations(self):
        self.check_device("cpu")

    def test_cuda_matches_dense_equations(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        self.check_device("cuda:0")


if __name__ == "__main__":
    unittest.main()
