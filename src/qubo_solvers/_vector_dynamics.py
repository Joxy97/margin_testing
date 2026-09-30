"""Shared integration and decoding for three vector relaxations."""
import torch
from ._dynamics import SpinObjective, finite, normal, publish, spins, unit


def search_vectors(solver, run, record, *, project=True, precession=0.):
    objective = SpinObjective(run)
    vector = normal(run, (*run.x.shape, 3))
    vector[..., 2] = vector[..., 2].abs()*spins(run)
    vector = unit(vector)
    for k in range(solver.max_steps):
        progress = (k+1)/max(solver.max_steps, 1)
        gradient = solver._gradient(vector, progress, objective)
        if project:
            tangent = gradient-vector*(vector*gradient).sum(-1, keepdim=True)
            drift = -tangent+precession*torch.linalg.cross(vector, gradient)
            vector = unit(vector+solver.time_step*drift)
        else:
            vector -= solver.time_step*gradient
        publish(run, vector[..., 2])
        run.iterations += 1
        record(k+1)
    finite(vector)
    return solver.max_steps
