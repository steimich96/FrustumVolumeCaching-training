
from typing import Callable

def _make_lazy_cuda_func(name: str) -> Callable:
    def call_cuda(*args, **kwargs):
        # pylint: disable=import-outside-toplevel
        from ._backend import _C

        return getattr(_C, name)(*args, **kwargs)

    return call_cuda


# occupancy grid
sample_occupancy_grid = _make_lazy_cuda_func("sample_occupancy_grid")

# distortion loss
distortion_loss = _make_lazy_cuda_func("distortion_loss")

# scan
inclusive_sum = _make_lazy_cuda_func("inclusive_sum")
exclusive_sum = _make_lazy_cuda_func("exclusive_sum")
inclusive_prod_forward = _make_lazy_cuda_func("inclusive_prod_forward")
inclusive_prod_backward = _make_lazy_cuda_func("inclusive_prod_backward")
exclusive_prod_forward = _make_lazy_cuda_func("exclusive_prod_forward")
exclusive_prod_backward = _make_lazy_cuda_func("exclusive_prod_backward")