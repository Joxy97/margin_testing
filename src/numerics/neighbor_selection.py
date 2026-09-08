"""Strongest-neighbor selection with index-stable ties in both array backends."""

import numpy


def numpyStrongestNeighbors(values: numpy.ndarray, count: int) -> numpy.ndarray:
    """Select largest values, resolving equal strengths by lowest column index."""
    threshold = numpy.partition(values, values.shape[1] - count, axis=1)[:, values.shape[1] - count]
    selected = values > threshold[:, None]
    ties = values == threshold[:, None]
    needed = count - selected.sum(axis=1)
    selected |= ties & (ties.cumsum(axis=1) <= needed[:, None])
    indices = numpy.nonzero(selected)[1].reshape(len(values), count)
    strength = numpy.take_along_axis(values, indices, axis=1)
    order = numpy.argsort(strength, axis=1, kind="stable")
    return numpy.take_along_axis(indices, order, axis=1)


def torchStrongestNeighbors(torch, values, count: int):
    """The same threshold/index rule without copying strengths to the host."""
    threshold = torch.topk(values, count, dim=1, sorted=False).values.amin(dim=1)
    selected = values > threshold[:, None]
    ties = values == threshold[:, None]
    needed = count - selected.sum(dim=1)
    selected |= ties & (ties.cumsum(dim=1) <= needed[:, None])
    indices = torch.nonzero(selected, as_tuple=True)[1].reshape(len(values), count)
    strength = values.gather(1, indices)
    order = torch.argsort(strength, dim=1, stable=True)
    return indices.gather(1, order)
