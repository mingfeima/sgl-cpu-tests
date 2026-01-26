import torch
import torch.nn.functional as F
from time import time

from utils import compare

import sgl_kernel
flash_attn_varlen_func = torch.ops.sgl_kernel.flash_attn_varlen_func

# set seed
torch.manual_seed(1234)


def flash_attn_varlen_ref(
    q, k, v,
    cu_seqlens_q,
    cu_seqlens_k,
    is_causal,
    enable_gqa,
):
    cu_q = cu_seqlens_q.tolist()
    cu_k = cu_seqlens_k.tolist()
    batch = len(cu_k) - 1

    # [T, H, D] -> [1, H, T, D]
    q, k, v = [x.unsqueeze(0).transpose(1, 2) for x in [q, k, v]]
    #print(q.size(), k.size(), v.size())

    B, H, T, D = q.shape
    out = torch.empty(B, H, T, v.size(-1), dtype=q.dtype)
    for b in range(batch):
        start_q, end_q = cu_q[b], cu_q[b + 1]
        start_k, end_k = cu_k[b], cu_k[b + 1]

        out[:, :, start_q:end_q, :] = F.scaled_dot_product_attention(
            q[:, :, start_q:end_q, :],
            k[:, :, start_k:end_k, :],
            v[:, :, start_k:end_k, :],
            is_causal=is_causal,
            enable_gqa=enable_gqa,
        )

    # [1, H, T, D] -> [T, H, D]
    return out.transpose(1, 2).squeeze(0)


def test_flash_attn_varlen(
    batch,
    max_seqlen_q,
    max_seqlen_k,
    num_heads,
    num_heads_kv,
    head_dim,
    head_dim_v,
    is_causal,
    is_varlen = True,
):
    dtype = torch.bfloat16



    # random seqlens for k and kv
    if is_varlen:
        seqlens_q = torch.randint(1, max_seqlen_q, (batch,), dtype=torch.int32)
        seqlens_k = torch.randint(1, max_seqlen_k, (batch,), dtype=torch.int32)
    else:
        seqlens_q = torch.Tensor([max_seqlen_q] * batch).to(dtype=torch.int32)
        seqlens_k = torch.Tensor([max_seqlen_k] * batch).to(dtype=torch.int32)

    cu_seqlens_q = torch.zeros((batch + 1,), dtype=torch.int32)
    cu_seqlens_k = torch.zeros((batch + 1,), dtype=torch.int32)
    cu_seqlens_q[1:] = torch.cumsum(seqlens_q, 0)
    cu_seqlens_k[1:] = torch.cumsum(seqlens_k, 0)

    #cu_seqlens_q = torch.tensor([0, 7], dtype=torch.int32)
    #cu_seqlens_k = torch.tensor([0, 38], dtype=torch.int32)

    print(seqlens_q, cu_seqlens_q)
    print(seqlens_k, cu_seqlens_k)

    sum_seqlen_q = seqlens_q.sum().item()
    sum_seqlen_k = seqlens_k.sum().item()
    q = torch.randn(sum_seqlen_q, num_heads, head_dim).to(dtype)
    k = torch.randn(sum_seqlen_k, num_heads_kv, head_dim).to(dtype)
    v = torch.randn(sum_seqlen_k, num_heads_kv, head_dim_v).to(dtype)

    #q.fill_(1)
    # trick for debugging flash attn
    #for i in range(sum_seqlen_k):
    #    k[i].fill_(i)
    #    v[i].fill_(i)

    out_ref = flash_attn_varlen_ref(
        q, k, v,
        cu_seqlens_q,
        cu_seqlens_k,
        is_causal=is_causal,
        enable_gqa= num_heads != num_heads_kv)

    out = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqlens_q.max().item(),
        seqlens_k.max().item(),
        is_causal)

    compare(out_ref, out, False)

test_flash_attn_varlen(1, 123, 45, 1, 1, 128, 96, is_causal=False, is_varlen=False)
test_flash_attn_varlen(4, 160, 60, 32, 4, 64, 94, is_causal=False) # 94 = 64 + 16 + 14
test_flash_attn_varlen(4, 160, 60, 32, 4, 64, 72, is_causal=True)  # 72 = 64 + 8
test_flash_attn_varlen(4, 160, 60, 32, 4, 64, 80, is_causal=True)  # 80 = 64 + 16
test_flash_attn_varlen(4, 160, 60, 32, 4, 64, 96, is_causal=True, is_varlen=False)
test_flash_attn_varlen(4, 160, 60, 32, 4, 64, 96, is_causal=False, is_varlen=False)

def bench_one_time(T, H, Hkv, K, is_causal=False):

    B = 6

    nwarmups = 10
    niters = 50

    enable_gqa = H != Hkv

    q = torch.randn(B*T, H, K).bfloat16()
    k = torch.randn(B*T, Hkv, K).bfloat16()
    v = torch.randn(B*T, Hkv, K).bfloat16()

    cu_seqlens = torch.arange(0,(B + 1) * T, step=T, dtype=torch.int32)


    qq, kk, vv = [x.unsqueeze(0).transpose(1, 2) for x in [q, k, v]]


    #for _ in range(nwarmups):
    #    out_ref = F.scaled_dot_product_attention(qq, kk, vv, is_causal=is_causal, enable_gqa=enable_gqa)

    t1 = time()
    #for _ in range(niters):
    #    out_ref = F.scaled_dot_product_attention(qq, kk, vv, is_causal=is_causal, enable_gqa=enable_gqa)
    t2 = time()

    q = q.transpose(0, 1).contiguous().transpose(0, 1)
    k = k.transpose(0, 1).contiguous().transpose(0, 1)
    v = v.transpose(0, 1).contiguous().transpose(0, 1)

    for _ in range(nwarmups):
        out = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, T, T, is_causal)

    t3 = time()
    for _ in range(niters):
        out = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, T, T, is_causal)
    t4 = time()

    #compare(out_ref.transpose(1, 2).squeeze(0), out, False)
    tt0 = (t2 - t1) / niters * 1000 #ms
    tt1 = (t4 - t3) / niters * 1000 #ms
    print(f"### T = {T}, H = {H}, Hkv = {Hkv}, K = {K}; torch.sdpa time {tt0:.3f} ms, flash_attn time {tt1:.3f} ms")


bench_one_time(8160, 6, 6, 72)

