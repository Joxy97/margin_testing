"""Energy identities and physical invariants, independent of optimum finding.

These checks run on CPU and available CUDA hardware. They exercise the
mathematical contracts underlying all five specialized physics backends.
"""

import itertools

import numpy as np
import pytest
import torch

from qubo_solvers import create_bqm_solver
from qubo_solvers.backends.altermagnet import TorchAltermagnetBQMSolver
from qubo_solvers.backends.phonon_exchange import TorchPhononExchangeBQMSolver
from qubo_solvers.backends.problem import QUBOProblem
from qubo_solvers.backends.supersymmetric import TorchSupersymmetricBQMSolver
from qubo_solvers.backends.torch_physics import PhysicsModel


PHYSICS_SOLVERS = (
    "lib_altermagnet", "lib_dynamical_geometry", "lib_geometric",
    "lib_phonon_exchange", "lib_supersymmetric",
)


def _source_problem():
    # Diagonal, reversed, duplicate and mixed-sign terms are deliberate.
    return QUBOProblem(
        [-1.5, .75, 2., -.5, .125],
        [0, 1, 0, 2, 3, 4, 1, 2], [1, 0, 1, 2, 4, 3, 2, 1],
        [.5, -.25, .75, -1.25, -2., .5, .625, -.125],
        offset=7.25, seedOffset=31,
    )


def _source_energy(bits):
    a, b, c, d, e = bits
    return 7.25 - 1.5*a + .75*b + .75*c - .5*d + .125*e + a*b + .5*b*c - 1.5*d*e


def _model(name, problem, device, dtype, **parameters):
    solver = create_bqm_solver(name, {"device": str(device)})
    options = solver._getParameters(dict(dtype=str(dtype).removeprefix("torch."),
        conditional_rounding=False, local_search_steps=0, **parameters))
    model = PhysicsModel(solver, problem, options, solver._size(problem.variableCount))
    return solver, model, options


@pytest.mark.parametrize("name", PHYSICS_SOLVERS)
@pytest.mark.parametrize("matrix_format", ["dense", "sparse"])
def test_normalized_ising_preserves_source_energy_differences(name, matrix_format, device, dtype):
    problem = _source_problem()
    _, model, _ = _model(name, problem, device, dtype, matrix_format=matrix_format)
    assignments = list(itertools.product((0, 1), repeat=problem.variableCount))
    bits = torch.tensor(assignments, device=device, dtype=dtype)
    # Padding has no source field or interaction. Arbitrary padded coordinates
    # must therefore not change the normalized objective.
    spins = torch.full((model.size, len(bits)), .37, dtype=dtype, device=device)
    spins[:problem.variableCount] = (2*bits-1).T
    normalized = model.energy(spins, model.product(spins)) * model.scale
    expected = torch.tensor([_source_energy(row) for row in assignments],
                            device=device, dtype=torch.float64)
    assert all(problem.energy(row) == _source_energy(row) for row in assignments)
    tolerance = 2e-6 if dtype == torch.float32 else 2e-13
    torch.testing.assert_close(normalized-normalized[0], expected-expected[0],
                               atol=tolerance, rtol=tolerance)


def test_supersymmetric_angular_hessian_matches_autograd(device):
    _, model, _ = _model("lib_supersymmetric", _source_problem(), device,
                         torch.float64, matrix_format="dense")
    theta = torch.linspace(-1.3, 1.7, 2*model.size, device=device,
                           dtype=torch.float64).reshape(model.size, 2).requires_grad_()
    tangent = torch.linspace(.2, 1.1, theta.numel(), device=device,
                             dtype=torch.float64).reshape_as(theta)
    confinement = .17
    x, y = theta.cos(), theta.sin()
    product = model.product(x)
    energy = model.energy(x, product).sum() + confinement*y.square().sum()
    gradient, = torch.autograd.grad(energy, theta, create_graph=True)
    expected, = torch.autograd.grad((gradient*tangent).sum(), theta)
    actual = TorchSupersymmetricBQMSolver._hessianProduct(
        model, x, y, product+model.h, tangent, confinement)
    torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-13)


@pytest.mark.parametrize("damping", [0., .2, .8])
def test_altermagnet_rational_update_preserves_norm_and_constant_field_descent(device, dtype, damping):
    generator = torch.Generator().manual_seed(812)
    spin = torch.randn((7, 3, 3), generator=generator, dtype=torch.float64)
    gradient = torch.randn((7, 3, 3), generator=generator, dtype=torch.float64)
    spin = (spin/spin.norm(dim=-1, keepdim=True)).to(device=device, dtype=dtype)
    gradient = gradient.to(device=device, dtype=dtype)
    tau = torch.tensor([.05, .2, .4], device=device, dtype=dtype)[None, :, None]
    updated = TorchAltermagnetBQMSolver._rational(torch, spin, gradient, tau, damping)
    tolerance = 3e-6 if dtype == torch.float32 else 3e-13
    torch.testing.assert_close(updated.norm(dim=-1), torch.ones_like(updated[..., 0]),
                               atol=tolerance, rtol=tolerance)
    old_energy = (spin.double()*gradient.double()).sum(-1)
    new_energy = (updated.double()*gradient.double()).sum(-1)
    assert torch.all(new_energy <= old_energy+tolerance)
    if damping == 0:
        torch.testing.assert_close(new_energy, old_energy, atol=tolerance, rtol=tolerance)
    else:
        assert torch.any(new_energy < old_energy-1e-4)


@pytest.mark.parametrize("metric", ["flat", "mixed", "ellipse"])
def test_geometric_guard_descends_fixed_potential_and_preserves_domain(device, dtype, metric):
    problem = _source_problem()
    solver, model, options = _model("lib_geometric", problem, device, dtype,
        steps=40, runs=3, candidate_interval=1, metric=metric, matrix_format="dense",
        confinement_start=.13, confinement_end=.13)
    energies, snapshots = [], []

    def collect(x):
        snapshots.append(x.clone())
        energies.append(model.energy(x, model.product(x))
                        + .13*(1-x.double().square()).sum(0))

    solver._run(model, problem, options, 0, 3, collect)
    # The extra final capture is bestX selected by the unconfined objective;
    # the accepted dynamics have their own fixed-potential descent contract.
    accepted_energy = torch.stack(energies[:-1])
    assert len(accepted_energy) == options["steps"]+1
    assert torch.all(accepted_energy[1:] <= accepted_energy[:-1]+2e-12)
    assert all(torch.isfinite(x).all() and (x.abs() <= 1+2e-6).all() for x in snapshots)


def test_dynamical_geometry_exact_rotations_preserve_charge_norm(device, dtype):
    solver = create_bqm_solver("lib_dynamical_geometry", {"device": str(device)})
    problem = _source_problem()
    result = solver.solve(problem, dict(steps=80, runs=3, run_batch_size=2,
        seed=37, dtype=str(dtype).removeprefix("torch."), time_step=.03,
        charge_norm=1.7, charge_coupling=1.4, eta=2., field_seed=.2,
        candidate_interval=8, conditional_rounding=False, local_search_steps=0,
        matrix_format="dense"))
    assert result.energy == _source_energy(result.sample)
    details = solver.lastDiagnostics[0]["trajectories"]
    assert [detail["runs"] for detail in details] == [2, 1]
    tolerance = 2e-5 if dtype == torch.float32 else 2e-12
    assert all(np.isfinite(detail["chargeNormError"])
               and detail["chargeNormError"] < tolerance for detail in details)


@pytest.mark.parametrize("coupling", [-.75, .75])
@pytest.mark.parametrize("oscillator_points", [12, 32])
def test_phonon_calibration_recovers_both_pair_coupling_signs(device, coupling, oscillator_points):
    # QUBO representation of J*s0*s1 plus a constant; its Ising fields vanish.
    problem = QUBOProblem([-2*coupling, -2*coupling], [0], [1],
                          [4*coupling], offset=3.25)
    solver, model, options = _model("lib_phonon_exchange", problem, device,
        torch.float64, matrix_format="dense", oscillator_points=oscillator_points)
    template = solver._preparePhonon(model, options)
    diagonal = template["diagonal"][0, 0]
    off = template["off"].reshape(()).expand(oscillator_points-1)
    hamiltonian = torch.diag_embed(diagonal)
    hamiltonian += torch.diag(off, diagonal=1)+torch.diag(off, diagonal=-1)
    ground_energies = torch.linalg.eigvalsh(hamiltonian)[:, 0]
    signs = template["signs"]
    expected = np.sign(coupling)*signs[:, 0]*signs[:, 1]
    torch.testing.assert_close(ground_energies, expected, atol=2e-11, rtol=2e-11)
    assert template["pairs"] == 1
    assert template["delta"] > 0


def test_phonon_flow_preserves_pair_wavefunction_norm(device, dtype):
    solver = TorchPhononExchangeBQMSolver(device=str(device))
    problem = _source_problem()
    result = solver.solve(problem, dict(steps=32, runs=3, run_batch_size=2,
        seed=47, dtype=str(dtype).removeprefix("torch."), time_step=.03,
        oscillator_points=12, candidate_interval=4, conditional_rounding=False,
        local_search_steps=0, matrix_format="dense"))
    assert result.energy == _source_energy(result.sample)
    tolerance = 3e-6 if dtype == torch.float32 else 3e-13
    for detail in solver.lastDiagnostics[0]["trajectories"]:
        assert detail["blockNormError"] < tolerance
        assert detail["terminalTransverseField"] == 0
