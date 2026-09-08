"""Weighted eigensystem with explicit space and eigenvector-sign policies."""

import numpy


def weightedEigensystem(values: numpy.ndarray, weights: numpy.ndarray, components: int,
                       *, observationSpace: bool, canonicalSigns: bool):
    mean = numpy.sum(weights[:, None] * values, axis=0)
    centered = values - mean
    wide = observationSpace
    if wide:
        weighted = numpy.sqrt(weights[:, None]) * centered
        matrix = weighted @ weighted.T
    else:
        matrix = centered.T @ (weights[:, None] * centered)
    eigenvalues, vectors = numpy.linalg.eigh(matrix)
    order = numpy.argsort(eigenvalues)[::-1]
    eigenvalues = numpy.maximum(eigenvalues[order], 0.0)
    selected = eigenvalues[:components]
    vectors = vectors[:, order[:components]]
    if wide:
        if numpy.any(selected <= numpy.finfo(float).eps):
            raise ValueError("requested PCA components include a zero-variance mode")
        vectors = (weighted.T @ vectors) / numpy.sqrt(selected)[None, :]
    total = float(eigenvalues.sum())
    if total <= 0:
        raise ValueError("PCA requires positive return variance")
    loadings = vectors.T.copy()
    if canonicalSigns:
        pivots = numpy.argmax(numpy.abs(loadings), axis=1)
        signs = numpy.sign(loadings[numpy.arange(components), pivots])
        signs[signs == 0] = 1
        loadings *= signs[:, None]
    return mean, centered, selected, selected / total, loadings
