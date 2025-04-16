import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import compare

import sgl_kernel

torch.manual_seed(1111)

def forward_native(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]

def run_single_test(shape, dtype, device):
    x = torch.randn(shape, dtype=dtype).to(device=device)

    d = x.shape[-1] // 2
    output_shape = x.shape[:-1] + (d,)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)

    out = torch.ops.sgl_kernel.silu_and_mul_cpu(x)
    ref_out = forward_native(x)

    compare(out, ref_out, False)

run_single_test([128, 22016], torch.bfloat16, "cpu")
run_single_test([129, 22016], torch.float16, "cpu")
