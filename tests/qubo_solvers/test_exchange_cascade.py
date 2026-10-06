"""Exchange physics, opt-in activation fluctuations, and capture cadence."""
from dataclasses import dataclass
from contextlib import nullcontext
from types import SimpleNamespace
import weakref

import pytest
import torch

from qubo_solvers import ExchangeCascade, QUBO
from qubo_solvers._dynamics import SpinObjective, normal, publish, spins, unit
from qubo_solvers.exchange_cascade import _ExchangeGraphBlock, _ExchangeWorkspace, _exchange_schedule
from qubo_solvers.observation import SolveInterrupted, observing
from qubo_solvers.solvers import _Run


def parameters(**updates):
    result = dict(k=1., xi=4., time_step=.1, field_damping=1.,
                  spin_mobility=1., order_mobility=1., max_spin_step=.2,
                  order_strength=4., orthogonality=1., order_seed=.05,
                  terminal_mass=2.)
    return dict(result, **updates)


@dataclass(frozen=True)
class LegacyExchange(ExchangeCascade):
    """Frozen original schedule/initialization for default compatibility."""

    def _search(self, run, record):
        obj = SpinObjective(run, normalize=False)
        n, batch = run.x.shape[1] + 1, run.x.shape[0]
        matrix = run.q.new_zeros((n, n))
        matrix[1:, 1:] = 2 * obj.j
        matrix[0, 1:] = obj.h
        matrix[1:, 0] = obj.h
        degree = matrix.abs().sum(1)
        scale = degree.max().clamp_min(1)
        matrix /= scale
        degree /= scale
        vectors = unit(normal(run, (n, batch, 3)))
        sign = torch.where((vectors[1:] * vectors[:1]).sum(-1) >= 0, 1., -1.)
        vectors[1:] *= (sign * spins(run).T)[..., None]
        work = _ExchangeWorkspace(torch, matrix, degree, vectors,
                                  parameters(k=self.k, xi=self.xi,
                                             time_step=self.time_step,
                                             terminal_mass=self.terminal_mass))
        freeze_step = min(self.max_steps - 1, int(.75 * self.max_steps))
        for step in range(self.max_steps):
            if step == freeze_step:
                work.freeze()
            progress = step / max(self.max_steps, 1)
            rho_a = run.q.new_tensor(max(-1., min(1., 2 * (progress - .2) / .25 - 1)))
            rho_b = run.q.new_tensor(max(-1., min(1., 2 * (progress - .45) / .3 - 1)))
            work.advance(rho_a, rho_b, terminal=step >= freeze_step)
            publish(run, 2 * work.decoded().to(run.q.dtype) - 1)
            run.iterations += 1
            record(step + 1)
        return self.max_steps


@pytest.mark.parametrize('steps', [0, 1, 7, 32])
def test_default_matches_original_complete_results(steps, device, dtype):
    q = QUBO(torch.tensor([[-1., .3, -.5], [.3, -.25, .4], [-.5, .4, .2]],
                          device=device, dtype=dtype))
    options = dict(restarts=5, batch_size=2, seed=23, history_interval=3)
    original = LegacyExchange(steps).solve(q, **options)
    actual = ExchangeCascade(steps).solve(q, **options)
    for name in ('final_assignments', 'final_energies', 'best_assignments',
                 'best_energies', 'iterations', 'termination_reasons',
                 'energy_history', 'history_iterations'):
        torch.testing.assert_close(getattr(actual, name), getattr(original, name), rtol=0, atol=0)


@pytest.mark.parametrize('steps,history_interval', [(0, 3), (1, 3), (13, 3), (13, 4), (20, 1)])
def test_sparse_capture_keeps_every_step_best_and_exact_histories(steps, history_interval, device, dtype):
    q = QUBO(torch.tensor([[-1., .5], [.5, -.7]], device=device, dtype=dtype))
    options = dict(restarts=4, batch_size=2, seed=11, history_interval=history_interval)
    every_step = ExchangeCascade(steps).solve(q, **options)
    sparse = ExchangeCascade(steps, candidate_interval=6).solve(q, **options)
    for name in ('final_assignments', 'best_assignments', 'best_energies',
                 'energy_history', 'history_iterations', 'iterations'):
        torch.testing.assert_close(getattr(sparse, name), getattr(every_step, name), rtol=0, atol=0)


def test_candidate_interval_changes_capture_only_and_final_is_always_captured():
    class Observer:
        def __init__(self):
            self.steps = []

        def poll(self):
            pass

        def capture(self, samples, raw=None, *, phase='checkpoint', iteration=None):
            self.steps.append(iteration)

    q = QUBO(torch.tensor([[-1., .5], [.5, -.7]]))
    observer = Observer()
    with observing(observer):
        sparse = ExchangeCascade(13, candidate_interval=5).solve(q, restarts=3, seed=11)
    full = ExchangeCascade(13).solve(q, restarts=3, seed=11)
    assert observer.steps == [0, 5, 10, 13]
    torch.testing.assert_close(sparse.best_energies, full.best_energies, rtol=0, atol=0)
    torch.testing.assert_close(sparse.best_assignments, full.best_assignments, rtol=0, atol=0)


def test_activation_seed_handles_tiny_zero_and_mature_vectors(device, dtype):
    work = _ExchangeWorkspace(torch, torch.zeros((2, 2), device=device, dtype=dtype),
                              torch.ones(2, device=device, dtype=dtype),
                              torch.zeros((2, 3, 3), device=device, dtype=dtype), parameters())
    tiny = 1e-30 if dtype == torch.float32 else 1e-200
    work.a.copy_(torch.tensor([[tiny, -2 * tiny, tiny], [0., 0., 0.], [.2, .3, .4]],
                             device=device, dtype=dtype))
    mature = work.a[2].clone()
    work.seed_order('a', 0)
    expected = torch.tensor([1., -2., 1.], device=device, dtype=dtype)
    expected *= .05 / expected.norm()
    torch.testing.assert_close(work.a[0], expected)
    torch.testing.assert_close(work.a[1], torch.tensor([.05, 0., 0.], device=device, dtype=dtype))
    torch.testing.assert_close(work.a[2], mature, rtol=0, atol=0)
    once = work.a.clone()
    work.seed_order('a', 0)
    torch.testing.assert_close(work.a, once)


def test_activation_seed_has_no_host_scalar_extraction(device, dtype):
    work = _ExchangeWorkspace(torch, torch.zeros((2, 2), device=device, dtype=dtype),
                              torch.ones(2, device=device, dtype=dtype),
                              torch.zeros((2, 3, 3), device=device, dtype=dtype), parameters())
    work.a.zero_()
    work.b.zero_()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        work.seed_order('a', 0)
        work.seed_order('b', 1)
    count = sum(event.count for event in profile.key_averages()
                if event.key == 'aten::_local_scalar_dense')
    assert count == 0


@pytest.mark.parametrize('freeze_fraction,expected_seeds,freeze_step',
                         [(0., [], 0), (.75, ['a', 'b'], 75), (1., ['a', 'b'], None)])
def test_activation_and_freeze_happen_once_at_requested_schedule(monkeypatch, freeze_fraction,
                                                               expected_seeds, freeze_step):
    seeded, frozen = [], []
    original_seed = _ExchangeWorkspace.seed_order
    original_freeze = _ExchangeWorkspace.freeze
    original_advance = _ExchangeWorkspace.advance

    def seed(work, name, axis):
        seeded.append(name)
        original_seed(work, name, axis)

    def freeze(work):
        frozen.append(getattr(work, 'test_step', 0))
        original_freeze(work)

    def advance(work, *args, **kwargs):
        original_advance(work, *args, **kwargs)
        work.test_step = getattr(work, 'test_step', 0) + 1

    monkeypatch.setattr(_ExchangeWorkspace, 'seed_order', seed)
    monkeypatch.setattr(_ExchangeWorkspace, 'freeze', freeze)
    monkeypatch.setattr(_ExchangeWorkspace, 'advance', advance)
    ExchangeCascade(100, freeze_fraction=freeze_fraction, order_seed_at_activation=True).solve(QUBO(-torch.eye(2)))
    assert seeded == expected_seeds
    assert frozen == ([] if freeze_step is None else [freeze_step])


def test_exchange_forces_and_order_update_match_same_physical_potential():
    dtype = torch.float64
    matrix = torch.tensor([[0., .4, -.3], [.4, 0., -.2], [-.3, -.2, 0.]], dtype=dtype)
    degree = matrix.abs().sum(1)
    x = unit(torch.tensor([[[.1, .2, .3]], [[.3, -.1, .2]], [[-.2, .2, .1]]], dtype=dtype))
    p = parameters(spin_mobility=0.)
    work = _ExchangeWorkspace(torch, matrix, degree, x.clone(), p)
    work.a.copy_(torch.tensor([[.1, .3, -.2]], dtype=dtype))
    work.b.copy_(torch.tensor([[-.2, .1, .25]], dtype=dtype))
    a, b = work.a.clone().requires_grad_(), work.b.clone().requires_grad_()
    rho_a, rho_b = x.new_tensor(.3), x.new_tensor(-.2)
    work.advance(rho_a, rho_b)
    z = work.z.detach()
    xx = x.clone().requires_grad_()
    laplacian = torch.diag(degree) - matrix
    force = p['k'] * torch.einsum('ij,jbd->ibd', laplacian, z)
    covariance = torch.bmm(z.permute(1, 2, 0), force.permute(1, 0, 2))
    strength = p['order_strength'] * max(float(degree.sum()) / 2, 1.)
    # Exchange Hamiltonian uses K=kI+xi(aa^T+bb^T), L=D_abs-A.
    kz = p['k'] * z + p['xi'] * ((z * a).sum(-1, keepdim=True) * a
                               + (z * b).sum(-1, keepdim=True) * b)
    field_energy = .5 * (force * kz).sum() - (force * xx).sum()
    aa, bb, ab = (a * a).sum(), (b * b).sum(), (a * b).sum()
    potential = field_energy + .25 * strength * ((aa - rho_a)**2 + (bb - rho_b)**2)
    potential += .5 * strength * p['orthogonality'] * ab**2
    grad_x, grad_a, grad_b = torch.autograd.grad(potential, (xx, a, b))
    torch.testing.assert_close(-grad_x, force)
    bound = strength * (3 * (aa + bb) + rho_a.abs() + rho_b.abs()
                        + p['orthogonality'] * (aa + bb))
    bound += p['xi'] * covariance.diagonal(dim1=-2, dim2=-1).sum().clamp_min(0)
    step = p['time_step'] * p['order_mobility'] / bound.clamp_min(1.)
    torch.testing.assert_close(work.a, a.detach() - step.detach() * grad_a)
    torch.testing.assert_close(work.b, b.detach() - step.detach() * grad_b)


@pytest.mark.parametrize('options', [dict(candidate_interval=0), dict(order_seed=0.),
    dict(order_seed_at_activation=1), dict(freeze_fraction=-.1), dict(freeze_fraction=1.1),
    dict(order_a_duration=0.), dict(order_b_start=.9, order_b_duration=.2),
    dict(order_a_start=float('nan')), dict(graph_block=-1), dict(graph_block=True),
    dict(spin_mobility=0.), dict(spin_mobility=float('inf'))])
def test_invalid_experimental_options(options):
    with pytest.raises(ValueError):
        ExchangeCascade(**options)


def test_graph_requires_cuda_and_rejects_unreliable_memory_contract():
    problem = QUBO(torch.eye(2))
    solver = ExchangeCascade(0, graph_block=25)
    with pytest.raises(ValueError, match='requires a CUDA problem'):
        solver.solve(problem)
    with pytest.raises(ValueError, match='does not support memory_limit_bytes'):
        solver.solve(problem, memory_limit_bytes=512*1024**2)
    with pytest.raises(ValueError, match='estimate_memory is unavailable'):
        solver.estimate_memory(problem)
    assert ExchangeCascade(0).estimate_memory(problem).total_bytes > 0


def test_optional_spin_mobility_is_forwarded_without_other_physics_changes(monkeypatch):
    seen = []
    original = _ExchangeWorkspace.__init__

    def init(work, torch, matrix, degree, vectors, p):
        seen.append(dict(p))
        original(work, torch, matrix, degree, vectors, p)

    monkeypatch.setattr(_ExchangeWorkspace, '__init__', init)
    problem = QUBO(-torch.eye(2))
    for mobility in [1., .25, 2.]:
        ExchangeCascade(3, spin_mobility=mobility).solve(problem, seed=11)
    assert seen == [parameters(spin_mobility=mobility) for mobility in [1., .25, 2.]]


@pytest.mark.parametrize('steps', [0, 1, 13, 10000])
def test_graph_schedule_has_exact_eager_python_values(steps, dtype):
    like = torch.empty(0, dtype=dtype)
    solver = ExchangeCascade(steps)
    actual = _exchange_schedule(solver, like)
    expected = []
    for step in range(steps):
        progress = step/max(steps, 1)
        expected.append(torch.stack([
            like.new_tensor(max(-1., min(1., 2*(progress-.2)/.25-1))),
            like.new_tensor(max(-1., min(1., 2*(progress-.45)/.3-1)))]))
    expected = torch.stack(expected) if steps else like.new_empty((0, 2))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('steps,freeze,reseed,history,block,mobility', [
    (0, .75, False, None, 25, 1.), (1, 0., False, None, 25, 1.),
    (13, 0., False, 5, 25, 1.), (103, .75, False, None, 25, 1.),
    (103, .75, True, None, 25, 1.), (103, 1., True, None, 25, 1.),
    (64, .75, True, 5, 25, 1.), (64, .6, True, 3, 7, .25),
    (80, .75, True, None, 8, 2.)])
def test_cuda_graph_complete_results_exactly_match_eager(steps, freeze, reseed, history,
                                                         block, mobility, dtype):
    q = QUBO(torch.tensor([[-1., .3, -.5], [.3, -.25, .4], [-.5, .4, .2]],
                          device='cuda', dtype=dtype), 7.)
    saved = q.Q.clone()
    options = dict(max_steps=steps, freeze_fraction=freeze, order_seed_at_activation=reseed,
                   candidate_interval=25, spin_mobility=mobility)
    starts = torch.tensor([[0, 0, 0], [1, 0, 1], [0, 1, 0], [1, 1, 0],
                           [1, 0, 0], [0, 0, 1], [1, 1, 1]], device='cuda', dtype=torch.int8)
    solve = dict(restarts=7, batch_size=3, seed=37, history_interval=history,
                 initial_assignments=starts)
    rng = torch.cuda.get_rng_state().clone()
    eager = ExchangeCascade(**options).solve(q, **solve)
    graphed = ExchangeCascade(**options, graph_block=block).solve(q, **solve)
    for name in ('final_assignments', 'final_energies', 'best_assignments', 'best_energies',
                 'iterations', 'termination_reasons', 'energy_history', 'history_iterations'):
        actual, expected = getattr(graphed, name), getattr(eager, name)
        if actual is None:
            assert expected is None
        else:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(q.Q, saved, rtol=0, atol=0)
    torch.testing.assert_close(torch.cuda.get_rng_state(), rng, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_graph_best_only_keeps_same_winner_and_owned_storage(dtype):
    q = QUBO(torch.tensor([[-1., .3, -.5], [.3, -.25, .4], [-.5, .4, .2]],
                          device='cuda', dtype=dtype), 7.)
    algorithm = dict(max_steps=103, candidate_interval=25, order_seed_at_activation=True,
                     graph_block=25)
    solve = dict(restarts=32, batch_size=32, seed=37)
    full = ExchangeCascade(**algorithm).solve(q, **solve)
    best = ExchangeCascade(**algorithm).solve(q, best_only=True, **solve)
    eager = ExchangeCascade(**dict(algorithm, graph_block=0)).solve(q, best_only=True, **solve)
    for name in ('best_assignment', 'best_energy', 'best_restart_index'):
        torch.testing.assert_close(getattr(best, name), getattr(full, name), rtol=0, atol=0)
        torch.testing.assert_close(getattr(best, name), getattr(eager, name), rtol=0, atol=0)
    assert best.best_assignments.untyped_storage().nbytes() == q.n_variables
    torch.testing.assert_close(q.energy(best.best_assignment), best.best_energy, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_graph_preparation_restores_state_before_deadline_and_replay_rejects_late_capture(dtype):
    from qubo_benchmark.runtime.worker import Observer

    problem = QUBO(torch.tensor([[-1., .3], [.3, -.4]], device='cuda', dtype=dtype))
    generator = torch.Generator(device='cuda').manual_seed(29)
    run = _Run(problem, 2, None, generator, torch.Generator().manual_seed(29))
    n = problem.n_variables+1
    matrix = problem.Q.new_zeros((n, n))
    degree = problem.Q.new_ones(n)
    vectors = unit(torch.randn((n, 2, 3), device='cuda', dtype=dtype, generator=generator))
    work = _ExchangeWorkspace(torch, matrix, degree, vectors, parameters())
    work.force.zero_()
    tensors = [work.x, work.z, work.a, work.b, work.force,
               run.x, run.field, run.best_x, run.best, run.iterations, run.reasons]
    saved = [value.clone() for value in tensors]
    energy = run.energies.clone()

    class Deadline:
        def poll(self):
            raise SolveInterrupted('deadline')

        def capture(self, *args, **kwargs):
            raise AssertionError('Graph preparation emitted a candidate')

    with observing(Deadline()), pytest.raises(SolveInterrupted, match='deadline'):
        _ExchangeGraphBlock(work, run, problem.Q.new_zeros((3, 2)), False)
    torch.cuda.synchronize()
    for actual, expected in zip(tensors, saved):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(run.energies, energy, rtol=0, atol=0)
    graph = _ExchangeGraphBlock(work, run, problem.Q.new_zeros((3, 2)), False)
    clock, events = [0], []
    observer = Observer(1., lambda message: events.extend(message.get('events', [])), clock=lambda: clock[0])
    with observing(observer):
        graph.replay(run, problem.Q.new_zeros((3, 2)))
        clock[0] = 2_000_000_000
        with pytest.raises(SolveInterrupted, match='deadline'):
            observer.capture(run.best_x, run.best, iteration=3)
    torch.cuda.synchronize()
    assert not events


@pytest.mark.parametrize('policy', ['no_observer', 'no_field', 'none_field', 'shared'])
def test_graph_capture_stream_selection_preserves_legacy_fallback(monkeypatch, policy):
    created, captured, streamed = [], [], []
    class Stream:
        def wait_stream(self, other):
            pass
    current_stream, shared_stream = Stream(), Stream()
    def make_stream(*, device):
        assert device == torch.device('cpu')
        stream = Stream()
        created.append(stream)
        return stream
    monkeypatch.setattr(torch.cuda, 'Stream', make_stream)
    monkeypatch.setattr(torch.cuda, 'CUDAGraph', lambda: object())
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: current_stream)
    monkeypatch.setattr(torch.cuda, 'stream', lambda stream: streamed.append(stream) or nullcontext())
    monkeypatch.setattr(torch.cuda, 'graph', lambda graph, *, stream: captured.append(stream) or nullcontext())
    observer = None if policy == 'no_observer' else SimpleNamespace(poll=lambda: None)
    if policy in ('none_field', 'shared'):
        observer.cuda_graph_stream = shared_stream if policy == 'shared' else None
    graphs = []
    for seed in (7, 8):
        problem = QUBO(-torch.eye(2))
        generator = torch.Generator().manual_seed(seed)
        run = _Run(problem, 2, None, generator, torch.Generator().manual_seed(seed))
        work = _ExchangeWorkspace(torch, torch.zeros((3, 3)), torch.ones(3),
                                  unit(torch.randn((3, 2, 3), generator=generator)), parameters())
        with observing(observer):
            graphs.append(_ExchangeGraphBlock(work, run, torch.empty((0, 2)), False))
    assert graphs[0].graph is not graphs[1].graph
    if policy == 'shared':
        assert created == [] and captured == streamed == [shared_stream, shared_stream]
    else:
        assert len(created) == 2 and created[0] is not created[1]
        assert captured == streamed == created


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_graph_shared_stream_reuses_infrastructure_across_fresh_seeds(monkeypatch, dtype):
    capture_stream = torch.cuda.Stream()
    captured, graph_refs, allocated = [], [], []
    original_graph, original_capture = torch.cuda.CUDAGraph, torch.cuda.graph
    def make_graph(*args, **kwargs):
        graph = original_graph(*args, **kwargs)
        graph_refs.append(weakref.ref(graph))
        return graph
    def capture(graph, *args, **kwargs):
        captured.append(kwargs['stream'])
        return original_capture(graph, *args, **kwargs)
    monkeypatch.setattr(torch.cuda, 'CUDAGraph', make_graph)
    monkeypatch.setattr(torch.cuda, 'graph', capture)
    q = QUBO(torch.tensor([[-1., .3, -.5], [.3, -.25, .4], [-.5, .4, .2]],
                          device='cuda', dtype=dtype), 7.)
    saved = q.Q.clone()
    options = dict(max_steps=13, freeze_fraction=1., candidate_interval=7)
    results = []
    for seed in (7, 8, 9, 7):
        eager = ExchangeCascade(**options).solve(q, restarts=4, batch_size=4, seed=seed)
        observer = SimpleNamespace(cuda_graph_stream=capture_stream, poll=lambda: None,
                                   capture=lambda *args, **kwargs: None)
        with observing(observer):
            graphed = ExchangeCascade(**options, graph_block=7).solve(q, restarts=4,
                                                                      batch_size=4, seed=seed)
        torch.cuda.synchronize()
        for name in ('final_assignments', 'final_energies', 'best_assignments',
                     'best_energies', 'iterations', 'termination_reasons'):
            torch.testing.assert_close(getattr(graphed, name), getattr(eager, name), rtol=0, atol=0)
        results.append(graphed.best_assignments.clone())
        allocated.append(torch.cuda.memory_allocated())
        assert all(ref() is None for ref in graph_refs), 'A mutable graph survived its fresh solve'
    assert len(captured) >= 8 and all(stream is capture_stream for stream in captured)
    torch.testing.assert_close(results[0], results[-1], rtol=0, atol=0)
    torch.testing.assert_close(q.Q, saved, rtol=0, atol=0)
    # Tiny result clones are retained above; a new 8.125 MiB cuBLAS workspace per
    # capture stream would exceed this generous fixed allowance after one seed.
    assert max(allocated[1:]) <= allocated[0]+2*1024**2
