import torch
from utils import compare
from typing import Optional
import torch.nn.functional as F

import sgl_kernel
convert_weight_packed = torch.ops.sgl_kernel.convert_weight_packed

from time import time

torch.manual_seed(1111)


def causal_conv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    return_final_states: bool = False,
    final_states_out: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
):

    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape
    if initial_states is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]
    if return_final_states:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(
            dtype_in
        )  # (batch, dim, width - 1)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return (out, None) if not return_final_states else (out, final_states_out)


def test_causal_conv1d(dim, seqlen, width, has_bias, dtype=torch.bfloat16):

    x = torch.randn(seqlen, dim).to(dtype)
    weight = torch.randn(dim, width).to(dtype)
    bias = torch.randn(dim) if has_bias else None

    # transpose
    x = x.transpose_(0, 1)
    print("@@@ x: ", x.size(), x.stride())
    print("@@@ weight: ", weight.size(), weight.stride())

    out_ref = causal_conv1d_ref(x, weight, bias)
    print(out_ref)

    #xx = torch.randn(1, 128, 256).to(dtype)
    #ww = torch.randn(256, 128, 4).to(dtype)
    #out = F.conv1d(xx, ww, None, padding=0)
    #print(out)


test_causal_conv1d(8192, 1024, 4, False)


def bench_causal_conv1d(dim, seqlen, width, dtype=torch.bfloat16):

    L = 20
    niters = 1000
    profile = True

    x = torch.randn(seqlen, dim).to(dtype)
    weight = torch.randn(dim, width).to(dtype)
    x = x.transpose_(0, 1).contiguous()

    inputs = [x.clone() for _ in range(L)]
    weights = [weight.clone() for _ in range(L)]

    # warmups
    for _ in range(niters // 100):
        for idx in range(L):
            out = causal_conv1d_ref(inputs[idx], weights[idx])

    t0 = time()
    with torch.autograd.profiler.profile(enabled=profile) as prof:
        for _ in range(niters):
            for idx in range(L):
                out = causal_conv1d_ref(inputs[idx], weights[idx])
        
    t1 = time()
    tt0 = (t1 - t0) / niters * 1000 / L # ms

    if profile:
        print(prof.key_averages().table(sort_by="cpu_time_total"))

    print(f"\n### causal_conv1d: oneDNN ref: {tt0:.3f} ms")


bench_causal_conv1d(8192, 1024, 4)

