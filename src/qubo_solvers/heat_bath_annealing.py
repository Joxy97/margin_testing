"""HeatBathAnnealing for unconstrained QUBO/Ising."""
from dataclasses import dataclass
import torch


from .simulated_annealing import SimulatedAnnealing
@dataclass(frozen=True)
class HeatBathAnnealing(SimulatedAnnealing):
    """D01 single-site Gibbs updates; one sweep visits each variable once."""

    def _search(self, run, record):
        for sweep in range(self.sweeps):
            t = sweep / max(self.sweeps-1, 1)
            temperature = self.start_temperature * (self.end_temperature/self.start_temperature)**t
            for column in torch.randperm(run.x.shape[1], generator=run.order_generator).tolist():
                direction = -2*run.x[:, column] if run.spin else 1-2*run.x[:, column]
                delta = direction*run.field[:, column]
                if not run.spin:
                    delta += run.q[column, column]
                uniform = torch.rand(delta.shape, dtype=run.q.dtype, device=run.q.device,
                                     generator=run.generator)
                run.flip_column(column, uniform < torch.sigmoid(-delta/temperature), delta)
            run.iterations += 1
            run.refresh()
            record(sweep+1)
        return self.sweeps

