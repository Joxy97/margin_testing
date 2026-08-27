#!/usr/bin/env python3
"""Extract exponentially weighted principal components from a CSV file."""

import argparse

import numpy as np
from sklearn.utils.extmath import randomized_svd


def ew_pca(z, k, decay=0.93, window=125):
    z = np.asarray(z, dtype=float)
    x = z[-window:]
    if z.ndim != 2 or not 1 <= k <= min(x.shape):
        raise ValueError("k must be between 1 and min(window observations, assets)")

    weights = decay ** np.arange(len(x) - 1, -1, -1)
    weights /= weights.sum()
    mean = weights @ x
    weighted = np.sqrt(weights[:, None]) * (x - mean)
    _, singular_values, components = randomized_svd(
        weighted, n_components=k, random_state=0
    )
    eigenvalues = singular_values**2
    scores = (z - mean) @ components.T
    return scores, components, eigenvalues, mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="CSV with a header and non-numeric first column")
    parser.add_argument("k", type=int)
    parser.add_argument("-o", "--output", default="pca_components.npz")
    parser.add_argument("--decay", type=float, default=0.93)
    parser.add_argument("--window", type=int, default=125)
    args = parser.parse_args()

    z = np.genfromtxt(args.input, delimiter=",", skip_header=1)[:, 1:]
    scores, components, eigenvalues, mean = ew_pca(
        z, args.k, args.decay, args.window
    )
    np.savez(
        args.output,
        scores=scores,
        components=components,
        eigenvalues=eigenvalues,
        mean=mean,
    )


if __name__ == "__main__":
    main()
