"""Dense portable exchange-field cascade adapted from margin_testing.

Keeps the analytic field propagator, staggered order controls and terminal-frame
freeze. Float64 is additionally supported. Optional CUDA graph blocks preserve
the per-step updates while reducing host launch overhead; eager is the default.
"""

from dataclasses import dataclass, replace
import math
import torch

from ._dynamics import IterativeSolver, SpinObjective, finite, normal, positive, publish, spins, unit
from .observation import current, observing, poll
from .solvers import _integer, _matrix

def _fieldPropagators(torch, a, b, k, xi, duration):
    """Evaluate exp(-duration K) and K^-1(I-exp(-duration K)) on device.

    Diagonalize the two-column Gram matrix analytically. This avoids a CUDA/CPU
    synchronization for a batched 3x3 eigh at every integration step. Vanishing
    Gram modes contribute zero; expm1 avoids cancellation for small timesteps.
    """
    u, v = math.sqrt(xi) * a, math.sqrt(xi) * b
    aa, bb, ab = (u * u).sum(-1), (v * v).sum(-1), (u * v).sum(-1)
    angle = .5 * torch.atan2(2 * ab, aa - bb)
    cosine, sine = angle.cos()[:, None], angle.sin()[:, None]
    modes = torch.stack((u * cosine + v * sine, -u * sine + v * cosine), dim=1)
    masses = (modes * modes).sum(-1)
    # A zero mode has a zero projector, including exactly zero/parallel seeds.
    projectors = modes[..., :, None] * modes[..., None, :]
    projectors = projectors / masses.clamp_min(torch.finfo(a.dtype).tiny)[..., None, None]
    eigenvalues = k + masses
    decay = torch.exp(-duration * eigenvalues)
    response = -torch.expm1(-duration * eigenvalues) / eigenvalues
    r0, q0 = math.exp(-duration * k), -math.expm1(-duration * k) / k
    identity = torch.eye(3, dtype=a.dtype, device=a.device)
    r = r0 * identity + ((decay - r0)[..., None, None] * projectors).sum(1)
    q = q0 * identity + ((response - q0)[..., None, None] * projectors).sum(1)
    return r, q, masses


class _ExchangeWorkspace:
    """One problem's shared interaction matrix and bounded trajectory state."""

    def __init__(self, torch, matrix, degree, spins, parameters):
        self.torch, self.matrix, self.degree, self.p = torch, matrix, degree, parameters
        self.x = spins  # (nodes, trajectories, Cartesian components)
        self.z = torch.zeros_like(spins)
        self.force = torch.empty_like(spins)
        width = spins.shape[1]
        self.a = spins.new_zeros((width, 3))
        self.b = spins.new_zeros((width, 3))
        self.a[:, 0].fill_(parameters['order_seed'])
        self.b[:, 1].fill_(parameters['order_seed'])
        # Match the original Python float arithmetic without extracting a CUDA
        # scalar. A zero-dimensional tensor retains scalar promotion semantics.
        self.orderScale = degree.sum().double().mul_(.5).clamp_min_(1.)
        self.orderScale.mul_(parameters['order_strength'])
        self.terminalPropagators = None
        self.preFreezeMasses = None
        self.frameFallback = None

    def feedback(self):
        torch = self.torch
        torch.mm(self.matrix, self.z.reshape(len(self.z), -1),
                 out=self.force.view(len(self.z), -1))
        self.force.neg_().add_(self.degree[:, None, None] * self.z).mul_(self.p['k'])
        return self.force

    def advance(self, rhoA, rhoB, terminal=False):
        torch, p = self.torch, self.p
        if terminal:
            r, q = self.terminalPropagators
        else:
            r, q, _ = _fieldPropagators(torch, self.a, self.b, p['k'], p['xi'],
                                      p['time_step'] / p['field_damping'])
        # Batched right multiplication in each trajectory, shared J via GEMM/SpMM.
        z, x = self.z.permute(1, 0, 2), self.x.permute(1, 0, 2)
        self.z.copy_((torch.bmm(z, r) + torch.bmm(x, q)).permute(1, 0, 2))
        force = self.feedback()
        tangent = force - self.x * (self.x * force).sum(-1, keepdim=True)
        delta = p['time_step'] * p['spin_mobility'] * tangent
        delta = delta / ((delta * delta).sum(-1, keepdim=True).sqrt()
                         / p['max_spin_step']).clamp_min(1.)
        self.x.add_(delta)
        self.x.div_(self.x.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(self.x.dtype).tiny))
        if not terminal:
            covariance = torch.bmm(self.z.permute(1, 2, 0), force.permute(1, 0, 2))
            aa = (self.a * self.a).sum(-1, keepdim=True)
            bb = (self.b * self.b).sum(-1, keepdim=True)
            ab = (self.a * self.b).sum(-1, keepdim=True)
            ca = torch.bmm(covariance, self.a[:, :, None]).squeeze(-1)
            cb = torch.bmm(covariance, self.b[:, :, None]).squeeze(-1)
            strength = self.orderScale
            gradA = strength * ((aa - rhoA) * self.a + p['orthogonality'] * ab * self.b) + p['xi'] * ca
            gradB = strength * ((bb - rhoB) * self.b + p['orthogonality'] * ab * self.a) + p['xi'] * cb
            # Conservative local curvature scale for explicit order updates.
            bound = strength * (3 * (aa + bb) + rhoA.abs() + rhoB.abs()
                                 + p['orthogonality'] * (aa + bb))
            bound += p['xi'] * covariance.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).clamp_min(0)
            step = p['time_step'] * p['order_mobility'] / bound.clamp_min(1.)
            self.a.add_(-step * gradA)
            self.b.add_(-step * gradB)

    def freeze(self):
        """Select an independent terminal frame; enforce both masses > k."""
        torch, p = self.torch, self.p
        _, _, self.preFreezeMasses = _fieldPropagators(
            torch, self.a, self.b, p['k'], p['xi'], p['time_step'] / p['field_damping'])
        tiny = 1e-6
        aNorm = self.a.norm(dim=-1, keepdim=True)
        defaultA = torch.zeros_like(self.a)
        defaultA[:, 0].fill_(1)
        axisA = torch.where(aNorm > tiny, self.a / aNorm.clamp_min(tiny), defaultA)
        perpendicular = self.b - (self.b * axisA).sum(-1, keepdim=True) * axisA
        bNorm = perpendicular.norm(dim=-1, keepdim=True)
        # Canonical least-aligned Cartesian axis, with lowest-index tie break.
        basis = torch.nn.functional.one_hot(axisA.abs().argmin(-1), 3).to(self.a.dtype)
        fallbackB = basis - (basis * axisA).sum(-1, keepdim=True) * axisA
        fallbackB /= fallbackB.norm(dim=-1, keepdim=True)
        axisB = torch.where(bNorm > tiny, perpendicular / bNorm.clamp_min(tiny), fallbackB)
        self.frameFallback = ((aNorm <= tiny) | (bNorm <= tiny)).squeeze(-1)
        radius = math.sqrt(p['terminal_mass'] / p['xi'])
        self.a.copy_(radius * axisA)
        self.b.copy_(radius * axisB)
        r, q, _ = _fieldPropagators(torch, self.a, self.b, p['k'], p['xi'],
                                  p['time_step'] / p['field_damping'])
        self.terminalPropagators = r, q

    def seed_order(self, name, axis):
        """Restore an activation fluctuation without replacing its direction.

        Negative order controls contract the homogeneous deterministic order
        equation. Long schedules can erase the initial fluctuation before the
        instability is reached. This optional finite fluctuation is injected
        once at activation; it is not a continuous floor on the dynamics.
        """
        torch = self.torch
        value = getattr(self, name)
        magnitude = value.abs().amax(-1, keepdim=True)
        # Scale before squaring so tiny but nonzero vectors keep their direction.
        denominator = torch.where(magnitude > 0, magnitude, torch.ones_like(magnitude))
        scaled = value / denominator
        scaled_norm = scaled.norm(dim=-1, keepdim=True)
        default = torch.zeros_like(value)
        default[:, axis].fill_(1)
        direction = torch.where(magnitude > 0,
                                scaled / scaled_norm.clamp_min(torch.finfo(value.dtype).tiny),
                                default)
        length = magnitude * scaled_norm
        value.copy_(torch.where(length < self.p['order_seed'],
                                self.p['order_seed'] * direction, value))

    def decoded(self):
        # Reference-spin gauge; dot == 0 selects binary 1. Isolated bits select 0.
        return (((self.x[1:] * self.x[:1]).sum(-1) >= 0)
                & (self.degree[1:, None] != 0)).T.contiguous()


def _exchange_schedule(solver, like):
    """Upload the exact legacy Python schedule values once for graph replay."""
    rows = []
    for step in range(solver.max_steps):
        progress = step / max(solver.max_steps, 1)
        rows.append([max(-1., min(1., 2*(progress-solver.order_a_start)/solver.order_a_duration-1)),
                     max(-1., min(1., 2*(progress-solver.order_b_start)/solver.order_b_duration-1))])
    return like.new_tensor(rows).reshape(solver.max_steps, 2)


class _ExchangeGraphBlock:
    """One solve's static CUDA graph; preparation never advances visible state."""

    def __init__(self, work, run, values, terminal):
        self.values = values.clone()
        self.graph = torch.cuda.CUDAGraph()
        observer = current()
        stream = getattr(observer, 'cuda_graph_stream', None)
        if stream is None:
            stream = torch.cuda.Stream(device=run.q.device)
        tensors = [work.x, work.z, work.a, work.b, work.force,
                   run.x, run.field, run.best_x, run.best, run.iterations, run.reasons]
        snapshots = [value.clone() for value in tensors]
        energy = run.energies.clone()

        def block():
            for index in range(len(self.values)):
                work.advance(self.values[index, 0], self.values[index, 1], terminal=terminal)
                publish(run, 2*work.decoded().to(run.q.dtype)-1)
                run.iterations.add_(1)

        # PyTorch requires side-stream warmup and stable tensor addresses.
        # The observer clock keeps running; only disposable preparation hooks
        # are disabled. Restore every mutated tensor before the real replay.
        stream.wait_stream(torch.cuda.current_stream(run.q.device))
        with observing(None), torch.cuda.stream(stream):
            for _ in range(3):
                block()
        torch.cuda.current_stream(run.q.device).wait_stream(stream)
        for value, saved in zip(tensors, snapshots):
            value.copy_(saved)
        run.energies.copy_(energy)
        stream.wait_stream(torch.cuda.current_stream(run.q.device))
        with observing(None), torch.cuda.graph(self.graph, stream=stream):
            block()
        self.energies = run.energies
        torch.cuda.current_stream(run.q.device).wait_stream(stream)
        for value, saved in zip(tensors, snapshots):
            value.copy_(saved)
        self.energies.copy_(energy)
        poll()

    def replay(self, run, values):
        self.values.copy_(values)
        self.graph.replay()
        # Each graph owns its final energy allocation. Other graphs use their
        # own private pools; select the correct output when switching graphs.
        run.energies = self.energies


@dataclass(frozen=True)
class ExchangeCascade(IterativeSolver):
    """Exchange cascade; iterations count integration steps.

    Legacy dynamics and observation cadence are the defaults. Activation
    seeding adds one finite fluctuation when each scheduled order control
    turns positive; feedback can shift the actual physical instability.
    A freeze fraction of one disables the terminal frame; zero freezes at once.
    Candidate intervals thin host observation only: binary scoring and best
    retention still happen on every step. Histories may require extra captures.
    graph_block=0 keeps eager execution. Positive blocks require CUDA; warmup
    and capture are charged to each fresh solve, with state restored before
    replay. Checkpoints, activation and freezing split the blocks. Graph mode
    rejects memory_limit_bytes and estimate_memory because CUDA capture pools
    have no reliable conservative tensor-storage bound here.
    """

    max_steps: int = 500
    time_step: float = .1
    k: float = 1.
    xi: float = 4.
    terminal_mass: float = 2.
    order_seed: float = .05
    order_seed_at_activation: bool = False
    order_a_start: float = .2
    order_a_duration: float = .25
    order_b_start: float = .45
    order_b_duration: float = .3
    freeze_fraction: float = .75
    candidate_interval: int = 1
    spin_mobility: float = 1.
    graph_block: int = 0

    def __post_init__(self):
        _integer('max_steps', self.max_steps, 0)
        _integer('candidate_interval', self.candidate_interval, 1)
        _integer('graph_block', self.graph_block, 0)
        for name in ('time_step', 'k', 'xi', 'terminal_mass', 'order_seed',
                     'order_a_duration', 'order_b_duration', 'spin_mobility'):
            positive(name, getattr(self, name))
        for name in ('order_a_start', 'order_b_start', 'freeze_fraction'):
            positive(name, getattr(self, name), zero=True)
            if getattr(self, name) > 1:
                raise ValueError(f'{name} must be <= 1')
        for name in ('a', 'b'):
            if getattr(self, f'order_{name}_start') + getattr(self, f'order_{name}_duration') > 1:
                raise ValueError(f'order_{name} ramp must finish by schedule fraction 1')
        if not isinstance(self.order_seed_at_activation, bool):
            raise ValueError('order_seed_at_activation must be a bool')
        if self.time_step > .25:
            raise ValueError('time_step must be <= .25')
        if self.terminal_mass <= self.k:
            raise ValueError('terminal_mass must exceed k')

    def solve(self, problem, *, restarts=1, initial_assignments=None, seed=None,
              history_interval=None, batch_size=None, best_only=False, memory_limit_bytes=None):
        if self.graph_block:
            if memory_limit_bytes is not None:
                raise ValueError('graph mode does not support memory_limit_bytes; CUDA capture pools are not bounded by the tensor estimate')
            if _matrix(problem).device.type != 'cuda':
                raise ValueError('CUDA graph execution requires a CUDA problem')
        solver = self
        if history_interval is not None:
            _integer('history_interval', history_interval, 1)
            solver = replace(self, candidate_interval=math.gcd(self.candidate_interval, history_interval))
        return IterativeSolver.solve(solver, problem, restarts=restarts,
                                     initial_assignments=initial_assignments, seed=seed,
                                     history_interval=history_interval, batch_size=batch_size,
                                     best_only=best_only, memory_limit_bytes=memory_limit_bytes)

    def estimate_memory(self, problem, *, restarts=1, batch_size=None,
                        best_only=False, history_interval=None):
        if self.graph_block:
            raise ValueError('estimate_memory is unavailable in graph mode; CUDA capture pools have no conservative bound')
        return IterativeSolver.estimate_memory(self, problem, restarts=restarts,
                                              batch_size=batch_size, best_only=best_only,
                                              history_interval=history_interval)

    def _search(self, run, record):
        obj = SpinObjective(run, normalize=False)
        n, batch = run.x.shape[1]+1, run.x.shape[0]
        matrix = run.q.new_zeros((n, n))
        matrix[1:, 1:] = 2*obj.j
        matrix[0, 1:] = obj.h
        matrix[1:, 0] = obj.h
        degree = matrix.abs().sum(1)
        scale = degree.max().clamp_min(1)
        matrix /= scale
        degree /= scale
        vectors = unit(normal(run, (n, batch, 3)))
        # Reflect into the hemisphere encoding the required initial assignment.
        sign = torch.where((vectors[1:]*vectors[:1]).sum(-1) >= 0, 1., -1.)
        vectors[1:] *= (sign*spins(run).T)[..., None]
        parameters = dict(k=self.k, xi=self.xi, time_step=self.time_step, field_damping=1.,
                          spin_mobility=self.spin_mobility, order_mobility=1., max_spin_step=.2,
                          order_strength=4., orthogonality=1., order_seed=self.order_seed,
                          terminal_mass=self.terminal_mass)
        work = _ExchangeWorkspace(torch, matrix, degree, vectors, parameters)
        observer = current()
        if observer is not None and getattr(observer, 'wall_schedule', False):
            return self._wall_search(work, run, record, observer)
        freeze_step = (self.max_steps if self.freeze_fraction == 1 else
                       min(self.max_steps-1, int(self.freeze_fraction*self.max_steps)))
        if self.graph_block:
            return self._graph_search(work, run, record, freeze_step)
        seeded_a = seeded_b = False
        for step in range(self.max_steps):
            if step == freeze_step:
                work.freeze()
            progress = step/max(self.max_steps, 1)
            rho_a = run.q.new_tensor(max(-1., min(1., 2*(progress-self.order_a_start)/self.order_a_duration-1)))
            rho_b = run.q.new_tensor(max(-1., min(1., 2*(progress-self.order_b_start)/self.order_b_duration-1)))
            if self.order_seed_at_activation and step < freeze_step:
                if not seeded_a and progress > self.order_a_start + .5*self.order_a_duration:
                    work.seed_order('a', 0)
                    seeded_a = True
                if not seeded_b and progress > self.order_b_start + .5*self.order_b_duration:
                    work.seed_order('b', 1)
                    seeded_b = True
            work.advance(rho_a, rho_b, terminal=step >= freeze_step)
            publish(run, 2*work.decoded().to(run.q.dtype)-1)
            run.iterations += 1
            if (step+1) % self.candidate_interval == 0 or step+1 == self.max_steps:
                record(step+1)
        finite(work.x, work.z, work.a, work.b)
        return self.max_steps

    def _wall_search(self, work, run, record, observer):
        """Opt-in elapsed-time controls; one state survives all checkpoints.

        Constant controls inside each replay block are explicit in this mode.
        The legacy step-scheduled graph/eager paths above remain unchanged.
        No max_steps-sized table is allocated for a wall-budget iteration ceiling.
        """
        graphs = {}
        seeded_a = seeded_b = terminal = False
        step = 0
        while step < self.max_steps:
            poll()
            progress = observer.schedule_fraction(0.)
            if not terminal and self.freeze_fraction<1 and progress>=self.freeze_fraction:
                work.freeze()
                terminal = True
            if self.order_seed_at_activation and not terminal:
                if not seeded_a and progress>self.order_a_start+.5*self.order_a_duration:
                    work.seed_order('a', 0)
                    seeded_a = True
                if not seeded_b and progress>self.order_b_start+.5*self.order_b_duration:
                    work.seed_order('b', 1)
                    seeded_b = True
            rho_a = max(-1., min(1., 2*(progress-self.order_a_start)/self.order_a_duration-1))
            rho_b = max(-1., min(1., 2*(progress-self.order_b_start)/self.order_b_duration-1))
            checkpoint = (step//self.candidate_interval+1)*self.candidate_interval
            end = min(step+(self.graph_block or 1), self.max_steps, checkpoint)
            if self.graph_block:
                length = end-step
                values = run.q.new_empty((length, 2))
                values[:, 0].fill_(rho_a)
                values[:, 1].fill_(rho_b)
                key = (terminal, length)
                if key not in graphs:
                    graphs[key] = _ExchangeGraphBlock(work, run, values, terminal)
                graphs[key].replay(run, values)
            else:
                work.advance(run.q.new_tensor(rho_a), run.q.new_tensor(rho_b), terminal=terminal)
                publish(run, 2*work.decoded().to(run.q.dtype)-1)
                run.iterations.add_(1)
            step = end
            if step % self.candidate_interval==0 or step==self.max_steps:
                record(step)
        finite(work.x, work.z, work.a, work.b)
        return self.max_steps

    def _graph_search(self, work, run, record, freeze_step):
        schedule = _exchange_schedule(self, run.q)
        activations = {}
        if self.order_seed_at_activation:
            for name, axis, onset in [('a', 0, self.order_a_start+.5*self.order_a_duration),
                                      ('b', 1, self.order_b_start+.5*self.order_b_duration)]:
                step = next((index for index in range(self.max_steps)
                             if index/max(self.max_steps, 1) > onset), self.max_steps)
                if step < freeze_step:
                    activations.setdefault(step, []).append((name, axis))
        boundaries = sorted({0, self.max_steps, max(0, freeze_step), *activations})
        graphs = {}
        step = 0
        while step < self.max_steps:
            poll()
            if step == freeze_step:
                work.freeze()
            for name, axis in activations.get(step, []):
                work.seed_order(name, axis)
            terminal = step >= freeze_step
            boundary = next(value for value in boundaries if value > step)
            checkpoint = (step//self.candidate_interval+1)*self.candidate_interval
            end = min(step+self.graph_block, self.max_steps, boundary, checkpoint)
            key = (terminal, end-step)
            if key not in graphs:
                graphs[key] = _ExchangeGraphBlock(work, run, schedule[step:end], terminal)
            graphs[key].replay(run, schedule[step:end])
            step = end
            if step % self.candidate_interval == 0 or step == self.max_steps:
                record(step)
        # Drain the queued work while graph pools and their outputs stay alive.
        finite(work.x, work.z, work.a, work.b)
        observer = current()
        if observer is not None and hasattr(observer, 'prepared'):
            observer.prepared(work.matrix, original_variables=run.x.shape[1],
                freeze_step=freeze_step, freeze_reached=self.max_steps > 0 and freeze_step < self.max_steps,
                terminal_steps=max(0, self.max_steps-max(0, freeze_step)),
                frame_fallback_count=int(work.frameFallback.sum().cpu()) if work.frameFallback is not None else None,
                prefreeze_masses=work.preFreezeMasses.cpu().tolist() if work.preFreezeMasses is not None else None,
                terminal_mass=self.terminal_mass, spin_mobility=self.spin_mobility)
        return self.max_steps
