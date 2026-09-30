from itertools import product

import pytest
import torch

from qubo_solvers import Ising, QUBO


def test_exhaustive_conversions(device, dtype):
    raw = torch.tensor([[2, -3, 1], [4, -2, 5], [-1, 2, 3]], device=device, dtype=dtype)
    x = torch.tensor(list(product([0, 1], repeat=3)), device=device, dtype=dtype)
    q = QUBO(raw, 2.5)
    expected = torch.stack([a @ raw @ a + 2.5 for a in x])
    torch.testing.assert_close(q.energy(x), expected)
    torch.testing.assert_close(q.to_ising().energy(2 * x - 1), expected)
    torch.testing.assert_close(q.to_ising().to_qubo().energy(x), expected)
    h = torch.tensor([1, -3, 2], device=device, dtype=dtype)
    s = 2 * x - 1
    i = Ising(raw, h, -1.5)
    expected = torch.stack([a @ raw @ a + h @ a - 1.5 for a in s])
    torch.testing.assert_close(i.energy(s), expected)
    torch.testing.assert_close(i.to_qubo().energy(x), expected)
    torch.testing.assert_close(i.to_qubo().to_ising().energy(s), expected)
    assert not i.J.diagonal().any()


def test_ownership_and_transfer(device, dtype):
    raw = torch.eye(2, device=device, dtype=dtype, requires_grad=True)
    h = torch.ones(2, device=device, dtype=dtype)
    offset = torch.tensor(3., device=device, dtype=dtype)
    q = QUBO(raw, offset)
    i = Ising(raw, h, offset)
    with torch.no_grad():
        raw.zero_()
        h.zero_()
        offset.zero_()
    assert q.Q.diagonal().sum() == 2
    assert i.h.sum() == 2
    assert q.offset == 3 and i.offset == 5
    assert not q.Q.requires_grad
    for problem in (q, i):
        moved = problem.to(device="cpu", dtype=torch.float64)
        assert moved is not problem
        coefficients = moved.Q if isinstance(moved, QUBO) else moved.J
        assert coefficients.device.type == "cpu" and coefficients.dtype == torch.float64


@pytest.mark.parametrize("matrix", [torch.ones(2), torch.ones(2, 3), torch.empty(0, 0),
                                    torch.tensor([[float('nan')]]), torch.tensor([[float('inf')]])])
def test_invalid_matrix(matrix):
    with pytest.raises(ValueError):
        QUBO(matrix)


def test_invalid_types_fields_offsets_and_assignments():
    with pytest.raises(TypeError):
        QUBO(torch.ones(2, 2, dtype=torch.int64))
    with pytest.raises(ValueError):
        Ising(torch.eye(2), torch.ones(3))
    with pytest.raises(ValueError):
        Ising(torch.eye(2), torch.ones(2, dtype=torch.float64))
    with pytest.raises(ValueError):
        QUBO(torch.eye(2), float("inf"))
    q = QUBO(torch.eye(2))
    for invalid in (torch.ones(3), torch.tensor([0.5, 1]), torch.tensor([float('nan'), 0])):
        with pytest.raises(ValueError):
            q.energy(invalid)
    assert q.energy(torch.tensor([True, False])).item() == 1
    assert q.energy(torch.ones(2, 3, 2)).shape == (2, 3)
