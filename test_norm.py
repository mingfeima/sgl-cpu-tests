import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from utils import compare

#from flashinfer.norm import rmsnorm, fused_add_rmsnorm
from sgl_kernel.ops._kernels import rmsnorm_cpu as rmsnorm
from sgl_kernel.ops._kernels import fused_add_rmsnorm_cpu as fused_add_rmsnorm

torch.manual_seed(1111)

def forward_native(
        x: torch.Tensor,
        weight: torch.Tensor,
        variance_epsilon: float = 1e-6,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + variance_epsilon)
        x = x.to(orig_dtype) * weight
        if residual is None:
            return x
        else:
            return x, residual

def run_single_test(shape, dtype, device="cuda"):
    
    x = torch.randn(shape, dtype=dtype).to(device=device)
    hidden_size = x.size(-1)
    weight = torch.randn(hidden_size, dtype=dtype).to(device=device)
    variance_epsilon = 1e-6

    print("\nTEST: rmsnorm")
    out = torch.empty_like(x)
    rmsnorm(out, x, weight, variance_epsilon)
    ref_out = forward_native(x, weight, variance_epsilon)

    compare(out, ref_out)

    print("\nTEST: fused_add_rmsnorm")
    # flashinfer writes x and residual inplaced
    ref_x = x.clone()

    residual = torch.randn(shape, dtype=dtype).to(device=device)
    ref_residual = residual.clone()

    fused_add_rmsnorm(x, residual, weight, variance_epsilon)

    ref_x, ref_residual = forward_native(ref_x, weight, variance_epsilon, ref_residual)

    compare(x, ref_x)
    compare(residual, ref_residual)

#run_single_test([4096, 4096], torch.bfloat16, "cuda")
run_single_test([1024, 4096], torch.bfloat16, "cpu")
run_single_test([1024, 4096 + 13], torch.float16, "cpu")




