"""ReplicaAnnealing for unconstrained QUBO/Ising."""
from dataclasses import dataclass
from . import observation
from ._dynamics import IterativeSolver, SpinObjective, finite, normal, positive, publish, spins
from .solvers import _integer


@dataclass(frozen=True)
class ReplicaAnnealing(IterativeSolver):
    """C17 general replica-coupled quartic OD flow, Euler (not ballistic SBQA).

    Each independent restart has replicas coupled by -coupling*sum(q_r*q_r+1).
    The coupling ramps linearly. Every layer is decoded each integration step.
    The final assignment is the best final layer under the original objective.
    """

    max_steps: int = 500
    replicas: int = 4
    time_step: float = .05
    coupling: float = .1
    penalty: float = 1.

    def __post_init__(self):
        _integer('max_steps', self.max_steps, 0)
        _integer('replicas', self.replicas, 3)
        positive('time_step', self.time_step)
        positive('coupling', self.coupling, zero=True)
        positive('penalty', self.penalty)

    def _extra_workspace(self, problem, batch):
        from .solvers import _matrix
        return super()._extra_workspace(problem, batch) + 12*batch*self.replicas*problem.n_variables*_matrix(problem).element_size()

    def _search(self, run, record):
        obj = SpinObjective(run)
        q = .1*spins(run)[:, None, :].repeat(1, self.replicas, 1)
        q += .01*normal(run, q.shape)
        for k in range(self.max_steps):
            t = observation.schedule_fraction((k+1)/max(self.max_steps, 1))
            gradient = (obj.gradient(q)+self.penalty*q*(q*q-1))/self.replicas
            gradient -= self.coupling*t*(q.roll(1, 1)+q.roll(-1, 1))
            q -= self.time_step*gradient
            candidates = (q >= 0).to(q.dtype).mul_(2).sub_(1)
            energies = obj.energy(candidates)
            if not run.spin:
                # Original QUBO and converted Ising scores differ only by a
                # constant mathematically, but cancellation can change ties.
                # Preserve the original-score incumbent and the historical
                # Ising-score final layer while avoiding one publish per layer.
                binary = (candidates + 1) * .5
                original = ((binary @ run.q) * binary).sum(-1)
                publish(run, candidates[run.rows, original.argmin(1)])
            # argmin keeps the first layer on ties, as the former ordered pass.
            publish(run, candidates[run.rows, energies.argmin(1)])
            run.iterations += 1
            record(k+1)
        finite(q)
        return self.max_steps

