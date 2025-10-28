import torch
from utils import compare
from typing import Optional
import torch.nn.functional as F

import sgl_kernel
causal_conv1d_weight_pack = torch.ops.sgl_kernel.causal_conv1d_weight_pack
causal_conv1d_fwd = torch.ops.sgl_kernel.causal_conv1d_fwd_cpu

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
    # TODO debug skip silu
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return (out, None) if not return_final_states else (out, final_states_out)


def test_weight_pack(dim, width):
    BLOCK_N = 32
    weight = torch.randn(dim, width).to(torch.bfloat16)
    packed_weight_ref = weight.view(dim // BLOCK_N, BLOCK_N, width // 2, 2).transpose_(1, 2).contiguous().view(dim, width)
    packed_weight = causal_conv1d_weight_pack(weight)

    res = torch.equal(packed_weight_ref, packed_weight)
    #print(packed_weight_ref)
    #print(packed_weight)
    print("test_weight_pack: ", res)

#test_weight_pack(8192, 4)
#test_weight_pack(32, 4)


def test_causal_conv1d(batch, dim, seqlen, width, has_bias=False, has_initial_state=False, dtype=torch.bfloat16):

    prepack = True

    x = torch.randn(batch, seqlen, dim).to(dtype)
    weight = torch.randn(dim, width).to(dtype)
    bias = torch.randn(dim).to(dtype) if has_bias else None

    if has_initial_state:
        initial_states = torch.randn(batch, dim, width - 1, dtype=dtype)
        has_initial_state_tensor = torch.ones(batch, dtype=torch.bool)
    else:
        initial_states = None
        has_initial_state_tensor = None

    packed_weight = causal_conv1d_weight_pack(weight) if prepack else weight

    x = x.transpose_(-1, -2)

    out_ref, final_states_ref = causal_conv1d_ref(
        x,
        weight,
        bias,
        initial_states,
        return_final_states=has_initial_state)

    def trans(a):
        if batch == 1:
            return a.transpose(-1, -2).contiguous().view(seqlen, dim)
        else:
            return a.transpose(-1, -2).contiguous().view(batch, seqlen, dim)

    def trans2(a):
        return a.transpose(-1, -2).contiguous().view(batch, width - 1, dim)

    out = causal_conv1d_fwd(
        x,
        packed_weight,
        bias,
        initial_states,
        None,
        None,
        has_initial_state_tensor,
        True,
        -1,
        prepack)

    compare(trans(out_ref), trans(out))
    if has_initial_state:
        compare(trans2(final_states_ref), trans2(initial_states))
    print("\n")


test_causal_conv1d(1, 8192, 1024, 4, has_bias=False)
test_causal_conv1d(1, 96, 36, 4, has_bias=False)
test_causal_conv1d(2, 64, 36, 4, has_bias=True)
test_causal_conv1d(1, 96, 36, 4, has_bias=True, has_initial_state=True)
test_causal_conv1d(2, 96, 2, 4, has_bias=True, has_initial_state=True)


PAD_SLOT_ID = -1

def test_causal_conv1d_varlen(batch, dim, max_length, width, has_bias=False, silu_activation=True, dtype=torch.bfloat16):

    padding = 3
    total_entries = batch + 3

    seqlens = torch.randint(1, max_length, (batch + 1,))
    seqlens[0] = 0
    # 1 or 2 must test
    seqlens[-2] = 2

    query_start_loc = torch.cumsum(seqlens, dim=0).to(torch.int32)

    seqlen = query_start_loc[-1].item()
    x = torch.randn(seqlen, dim, dtype=dtype).transpose_(-1, -2)
    weight = torch.randn(dim, width, dtype=dtype)
    bias = torch.randn(dim, dtype=dtype) if has_bias else None

    activation = None if not silu_activation else "silu"

    final_states = torch.randn(total_entries, dim, width - 1, dtype=dtype)
    final_states_ref = final_states.clone()

    has_initial_states = torch.randint(0, 2, (batch,), dtype=torch.bool).fill_(False)
    state_indices = torch.randperm(total_entries, dtype=torch.int32)[:batch]

    out_ref = []
    out_ref_b = []

    return_final_states = final_states is not None

    splits = torch.split(x, seqlens[1:].tolist(), dim=1)
    for i, x_s in enumerate(splits):
        out_ref_b.append(
            causal_conv1d_ref(
                x_s.unsqueeze(0),
                weight,
                bias,
                activation=activation,
                return_final_states=return_final_states,
                final_states_out=(
                    final_states_ref[state_indices[i]].unsqueeze(0)
                    if return_final_states
                    else None),
                initial_states=(
                    final_states_ref[state_indices[i]].unsqueeze(0)
                    if has_initial_states[i]
                    else None)
            )
        )
    out_ref.append(torch.cat([t[0] for t in out_ref_b], dim=2))
    out_ref_tensor = torch.cat(out_ref, dim=0)

    out = causal_conv1d_fwd(
        x,
        weight,
        bias,
        final_states,
        query_start_loc,
        state_indices,
        has_initial_states,
        activation in ["silu"],
        PAD_SLOT_ID,
        False)

    compare(out_ref_tensor.transpose(-1, -2), out.transpose(-1, -2))
    compare(final_states_ref, final_states)


test_causal_conv1d_varlen(11, 96, 66, 4, silu_activation=False)


def bench_causal_conv1d(batch, dim, seqlen, width, dtype=torch.bfloat16):

    L = 20 if batch < 4 else 1
    niters = 200 if batch < 4 else 50
    profile = False

    x = torch.randn(batch, seqlen, dim).to(dtype)
    weight = torch.randn(dim, width).to(dtype)
    x = x.transpose_(-1, -2)

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

    prepack = True
    t2 = time()
    for _ in range(niters):
        for idx in range(L):
            out = causal_conv1d_fwd(
                inputs[idx],
                weights[idx],
                None,
                None,
                None,
                None,
                None,
                False,
                -1,
                prepack)
    t3 = time()
    tt1 = (t3 - t2) / niters * 1000 / L # ms

    if profile:
        print(prof.key_averages().table(sort_by="cpu_time_total"))

    print(f"\n### causal_conv1d: oneDNN ref: {tt0:.3f} ms; opt: {tt1:.3f} ms")


bench_causal_conv1d(1, 8192, 1024, 4)
bench_causal_conv1d(128, 8192, 1024, 4)

