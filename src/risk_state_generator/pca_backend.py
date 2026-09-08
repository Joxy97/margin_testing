"""Interchangeable numerical backends for exponentially weighted PCA."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy


@dataclass(frozen=True)
class PCAFit:
    lambdas: numpy.ndarray
    explained: numpy.ndarray
    loadings: numpy.ndarray
    factors: numpy.ndarray
    pcaMean: numpy.ndarray
    residuals: numpy.ndarray
    residualScale: numpy.ndarray
    maxAbsoluteZ: numpy.ndarray


class PCABackend(ABC):
    @abstractmethod
    def fit(self, values: numpy.ndarray, weights: numpy.ndarray, components: int) -> PCAFit:
        """Fit standardized returns, returning host arrays for risk generation."""

    @staticmethod
    def validate(values: numpy.ndarray, weights: numpy.ndarray, components: int) -> None:
        if not isinstance(values, numpy.ndarray) or not isinstance(weights, numpy.ndarray):
            raise TypeError("PCA values and weights must be NumPy arrays")
        if values.ndim != 2 or len(values) < 2:
            raise ValueError("PCA requires at least two return observations")
        if not 1 <= components <= min(values.shape):
            raise ValueError(f"components must be between 1 and {min(values.shape)}, inclusive")
        if weights.shape != (len(values),):
            raise ValueError("weights must contain one value per observation")
        if (not numpy.isfinite(values).all() or not numpy.isfinite(weights).all()
                or numpy.any(weights < 0) or not numpy.isclose(weights.sum(), 1.0)):
            raise ValueError("PCA requires finite values and normalized nonnegative weights")


class NumpyPCABackend(PCABackend):
    def fit(self, values: numpy.ndarray, weights: numpy.ndarray, components: int) -> PCAFit:
        self.validate(values, weights, components)
        from numerics.weighted_pca import weightedEigensystem
        mean, centered, selected, explained, loadings = weightedEigensystem(
            values, weights, components, observationSpace=values.shape[1] > len(values),
            canonicalSigns=True)
        factors = centered @ loadings.T
        residuals = values - (mean + factors @ loadings)
        residual_mean = numpy.sum(weights[:, None] * residuals, axis=0)
        scale = numpy.sqrt(numpy.sum(weights[:, None] * (residuals - residual_mean)**2, axis=0))
        return PCAFit(selected, explained, loadings, factors, mean,
                      residuals, scale, numpy.max(numpy.abs(values), axis=0))


@dataclass(frozen=True)
class TorchPCAFit:
    """Adapter-owned tensors; host consumers explicitly materialize PCAFit."""

    lambdas: Any
    explained: Any
    loadings: Any
    factors: Any
    pcaMean: Any
    residuals: Any
    residualScale: Any
    maxAbsoluteZ: Any

    def toHost(self) -> PCAFit:
        return PCAFit(*(value.cpu().numpy().copy() for value in vars(self).values()))


@dataclass(frozen=True)
class TorchPCABackend(PCABackend):
    """Float64 PCA; Torch is imported only when this backend is used."""

    device: str = "auto"

    def fit(self, values: numpy.ndarray, weights: numpy.ndarray, components: int) -> PCAFit:
        return self.fitResident(values, weights, components).toHost()

    def fitResident(self, values: numpy.ndarray, weights: numpy.ndarray, components: int) -> TorchPCAFit:
        self.validate(values, weights, components)
        import torch

        device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu")
            if self.device == "auto" else self.device
        )
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("Torch PCA supports CPU and CUDA/ROCm devices")
        with torch.inference_mode(), torch.profiler.record_function("margin.pca"):
            x = torch.as_tensor(numpy.ascontiguousarray(values), dtype=torch.float64, device=device)
            w = torch.as_tensor(numpy.ascontiguousarray(weights), dtype=torch.float64, device=device)[:, None]
            mean = (w * x).sum(dim=0)
            centered = x - mean
            wide = x.shape[1] > x.shape[0]
            if wide:
                weighted = w.sqrt() * centered
                matrix = weighted @ weighted.T
            else:
                matrix = centered.T @ (w * centered)
            eigenvalues, vectors = torch.linalg.eigh(matrix)
            eigenvalues = eigenvalues.flip(0).clamp_min(0)
            selected = eigenvalues[:components]
            vectors = vectors.flip(1)[:, :components]
            if wide:
                if bool((selected <= torch.finfo(torch.float64).eps).any()):
                    raise ValueError("requested PCA components include a zero-variance mode")
                vectors = (weighted.T @ vectors) / selected.sqrt()[None, :]
            total = eigenvalues.sum()
            if float(total) <= 0:
                raise ValueError("PCA requires positive return variance")
            loadings = vectors.T.contiguous()
            pivots = loadings.abs().argmax(dim=1)
            signs = loadings[torch.arange(components, device=device), pivots].sign()
            signs.masked_fill_(signs == 0, 1)
            loadings *= signs[:, None]
            factors = centered @ loadings.T
            residuals = x - (mean + factors @ loadings)
            residual_mean = (w * residuals).sum(dim=0)
            scale = (w * (residuals - residual_mean).square()).sum(dim=0).sqrt()
            results = (selected, selected / total, loadings, factors, mean,
                       residuals, scale, x.abs().amax(dim=0))
            return TorchPCAFit(*results)


@dataclass(frozen=True)
class PCABackendConfig:
    type: str = "numpy"
    device: str = "auto"

    def __post_init__(self) -> None:
        if not isinstance(self.type, str) or not isinstance(self.device, str):
            raise TypeError("PCA backend type and device must be strings")
        if self.type not in {"numpy", "torch"}:
            raise ValueError("PCA backend type must be numpy or torch")
        if self.device not in {"auto", "cpu", "cuda"} and not (
            self.device.startswith("cuda:") and self.device[5:].isdigit()
        ):
            raise ValueError("PCA device must be auto, cpu, cuda, or cuda:<index>")
        if self.type == "numpy" and self.device not in {"auto", "cpu"}:
            raise ValueError("NumPy PCA requires a CPU device")

    def createBackend(self) -> PCABackend:
        return NumpyPCABackend() if self.type == "numpy" else TorchPCABackend(self.device)
