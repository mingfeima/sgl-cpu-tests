import torch
from utils import compare
from typing import Optional
import torch.nn.functional as F

import sgl_kernel
causal_conv1d_weight_pack = torch.ops.sgl_kernel.causal_conv1d_weight_pack
causal_conv1d_update = torch.ops.sgl_kernel.causal_conv1d_update_cpu

from time import time

torch.manual_seed(1111)


def causal_conv1d_update_ref(
    x, conv_state, weight, bias=None, activation=None, cache_seqlens=None
):
    x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]

    x_new = torch.cat([conv_state, x], dim=-1)
    conv_state.copy_(x_new[:, :, -state_len:])
    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=dim)[
        :, :, -seqlen:
    ]

    out = out.squeeze(-1)
    return out if activation is None else F.silu(out)


def test_causal_conv1d_update(batch, dim, width, has_bias=False, has_silu=True, dtype=torch.bfloat16):

    prepack = True
    activation = "silu" if has_silu else None

    x = torch.randn(batch, dim).to(dtype)
    conv_state = torch.randn(batch, dim, width - 1, dtype=dtype)
    weight = torch.randn(dim, width).to(dtype)
    bias = torch.randn(dim).to(dtype) if has_bias else None

    #for i in range(width - 1):
    #    conv_state[:, :, i].fill_(i)
    
    #x.fill_(99)

    packed_weight = causal_conv1d_weight_pack(weight) if prepack else weight

    conv_state_ref = conv_state.clone()
    out_ref = causal_conv1d_update_ref(
        x, conv_state_ref, weight, bias, activation=activation
    )

    cache_seqlens = None
    conv_state_indices = None
    out = causal_conv1d_update(
        x,
        conv_state,
        packed_weight,
        bias,
        has_silu,
        cache_seqlens=cache_seqlens,
        conv_state_indices=conv_state_indices,
        pad_slot_id=-1,
        is_vnni=prepack)

    #print("@@@ out_ref: ", out_ref, out_ref.size())
    #print("@@@ out: ", out, out.size())

    #print("@@@ conv_state_ref: ", conv_state_ref.transpose(-1, -2), conv_state_ref.size())
    #print("@@@ conv_state: ", conv_state.transpose(-1, -2), conv_state.size(), conv_state.stride())
    compare(out_ref, out)
    compare(conv_state_ref, conv_state)
    print("\n")


test_causal_conv1d_update(2, 96, 4, has_bias=False)


def test_causal_conv1d_update_with_batch_gather(
    batch, dim, width, has_bias=False, has_silu=True, dtype=torch.bfloat16
):
    prepack = True
    activation = "silu" if has_silu else None

    padding = 3
    total_entries = batch + 3

    x = torch.randn(batch, dim).to(dtype=dtype)

    conv_state_indices = torch.randperm(total_entries)[:batch].to(dtype=torch.int32)
    conv_state = torch.randn(total_entries, dim, width - 1, dtype=dtype)

    weight = torch.randn(dim, width).to(dtype=dtype)
    bias = torch.randn(dim).to(dtype=dtype) if has_bias else None
    conv_state_ref = conv_state[conv_state_indices, :]

    packed_weight = causal_conv1d_weight_pack(weight) if prepack else weight

    cache_seqlens = None
    out = causal_conv1d_update(
        x,
        conv_state,
        packed_weight,
        bias,
        has_silu,
        cache_seqlens=cache_seqlens,
        conv_state_indices=conv_state_indices,
        pad_slot_id=-1,
        is_vnni=prepack
    )

    out_ref = causal_conv1d_update_ref(
        x, conv_state_ref, weight, bias, activation=activation
    )

    compare(out_ref, out)
    compare(conv_state_ref, conv_state[conv_state_indices, :])
    print("\n")


test_causal_conv1d_update_with_batch_gather(4, 96, 4, has_bias=False, has_silu=True)
