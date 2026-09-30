"""Dense quadratic objectives. Stored tensors are read-only by convention."""

from dataclasses import dataclass

import torch
from torch import Tensor


def _coefficient(value: Tensor, *, copy: bool = True) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError("coefficients must be PyTorch tensors")
    if value.layout != torch.strided:
        raise ValueError("coefficients must be dense tensors")
    if value.device.type not in ("cpu", "cuda"):
        raise ValueError("only CPU and CUDA devices are supported")
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError("coefficients must use float32 or float64")
    if not torch.isfinite(value).all():
        raise ValueError("coefficients must be finite")
    return value.detach().clone() if copy else value.detach()


def _matrix(value: Tensor) -> Tensor:
    value = _coefficient(value, copy=False)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or not value.shape[0]:
        raise ValueError("coefficient matrix must be nonempty and square")
    result = value / 2
    result.add_(value.T, alpha=0.5)
    return result


def _offset(value: float | Tensor, like: Tensor) -> Tensor:
    value = torch.as_tensor(value, dtype=like.dtype, device=like.device)
    if value.ndim != 0 or not torch.isfinite(value):
        raise ValueError("offset must be a finite scalar")
    return value.detach().clone()


def _assignments(value: Tensor, like: Tensor, spin: bool, *, cast: bool = True) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError("assignments must be PyTorch tensors")
    if value.layout != torch.strided or value.is_complex():
        raise ValueError("assignments must be dense and real")
    if value.ndim < 1 or value.shape[-1] != like.shape[0]:
        raise ValueError("assignment last dimension must equal the variable count")
    if value.device != like.device:
        raise ValueError("assignments and problem must be on the same device")
    if not ((value == (-1 if spin else 0)) | (value == 1)).all():
        raise ValueError("assignments contain values outside the problem domain")
    return value.to(dtype=like.dtype) if cast else value


def _owned(cls, **values):
    """Build from internally normalized, exclusively owned coefficient tensors."""
    instance = object.__new__(cls)
    for name, value in values.items():
        object.__setattr__(instance, name, value)
    return instance


@dataclass(frozen=True, eq=False)
class QUBO:
    """E(x) = xᵀQx + offset for binary x; Q is symmetrized on construction."""

    Q: Tensor
    offset: float | Tensor = 0.0

    def __post_init__(self) -> None:
        matrix = _matrix(self.Q)
        object.__setattr__(self, "Q", matrix)
        object.__setattr__(self, "offset", _offset(self.offset, matrix))

    @property
    def n_variables(self) -> int:
        return self.Q.shape[0]

    def energy(self, assignments: Tensor) -> Tensor:
        """Evaluate shape (..., n), returning shape (...)."""
        x = _assignments(assignments, self.Q, spin=False)
        return self._energy(x)

    def _energy(self, x: Tensor) -> Tensor:
        return ((x @ self.Q) * x).sum(dim=-1) + self.offset

    def to(self, *, device=None, dtype=None) -> "QUBO":
        matrix = _coefficient(self.Q.to(device=device, dtype=dtype, copy=True), copy=False)
        return _owned(QUBO, Q=matrix, offset=_offset(self.offset, matrix))

    def to_ising(self) -> "Ising":
        matrix = self.Q / 4
        field = self.Q.sum(dim=1) / 2
        offset = _offset(self.offset + self.Q.sum() / 4 + matrix.diagonal().sum(), matrix)
        matrix.fill_diagonal_(0)
        _coefficient(field, copy=False)
        return _owned(Ising, J=matrix, h=field, offset=offset)


@dataclass(frozen=True, eq=False)
class Ising:
    """E(s) = sᵀJs + hᵀs + offset; symmetric off-diagonals count twice."""

    J: Tensor
    h: Tensor
    offset: float | Tensor = 0.0

    def __post_init__(self) -> None:
        matrix = _matrix(self.J)
        field = _coefficient(self.h)
        if field.shape != (matrix.shape[0],):
            raise ValueError("h must have shape (n_variables,)")
        if field.device != matrix.device or field.dtype != matrix.dtype:
            raise ValueError("J and h must have matching device and dtype")
        offset = _offset(self.offset, matrix) + matrix.diagonal().sum()
        matrix.fill_diagonal_(0)
        object.__setattr__(self, "J", matrix)
        object.__setattr__(self, "h", field)
        object.__setattr__(self, "offset", _offset(offset, matrix))

    @property
    def n_variables(self) -> int:
        return self.J.shape[0]

    def energy(self, assignments: Tensor) -> Tensor:
        """Evaluate shape (..., n), returning shape (...)."""
        s = _assignments(assignments, self.J, spin=True)
        return self._energy(s)

    def _energy(self, s: Tensor) -> Tensor:
        return ((s @ self.J) * s).sum(dim=-1) + (s * self.h).sum(dim=-1) + self.offset

    def to(self, *, device=None, dtype=None) -> "Ising":
        matrix = _coefficient(self.J.to(device=device, dtype=dtype, copy=True), copy=False)
        field = _coefficient(self.h.to(device=device, dtype=dtype, copy=True), copy=False)
        return _owned(Ising, J=matrix, h=field, offset=_offset(self.offset, matrix))

    def to_qubo(self) -> QUBO:
        matrix = 4 * self.J
        matrix.diagonal().copy_(2 * self.h - 4 * self.J.sum(dim=1))
        _coefficient(matrix, copy=False)
        return _owned(QUBO, Q=matrix,
                      offset=_offset(self.offset + self.J.sum() - self.h.sum(), matrix))


Problem = QUBO | Ising
