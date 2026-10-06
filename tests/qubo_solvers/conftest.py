import pytest
import torch


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def device(request):
    return torch.device(request.param)


@pytest.fixture(params=[torch.float32, torch.float64])
def dtype(request):
    return request.param


@pytest.fixture(autouse=True, scope="session")
def small_tensor_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)
