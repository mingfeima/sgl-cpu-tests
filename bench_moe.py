import torch
from sgl_kernel.common_ops import fused_experts_cpu as fused_experts
from sgl_kernel.common_ops import convert_weight_packed

from time import time

torch.manual_seed(1111)

BLOCK_N, BLOCK_K = 128, 128

def run_single_test(M, N, K, E, topk, dtype=torch.bfloat16):

    score = torch.randn(M, E).to(dtype=dtype)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_ids = topk_ids.to(torch.int32)

    prepack = True
    inplace = True

    input = torch.randn(M, K).to(dtype=dtype)
    w1 = torch.randn(E, 2 * N, K).to(dtype=dtype)
    w2 = torch.randn(E, K, N).to(dtype=dtype)

    packed_w1 = convert_weight_packed(w1)
    packed_w2 = convert_weight_packed(w2)

    # int8
    w1_int8 = torch.randn(E, 2 * N, K).to(dtype=torch.int8)
    w2_int8 = torch.randn(E, K, N).to(dtype=torch.int8)

    packed_w1_int8 = convert_weight_packed(w1_int8)
    packed_w2_int8 = convert_weight_packed(w2_int8)

    w1_s = torch.rand(E, 2 * N)
    w2_s = torch.rand(E, K)

    # fp8
    w1_fp8 = torch.randn(E, 2 * N, K).to(torch.float8_e4m3fn)
    w2_fp8 = torch.randn(E, K, N).to(torch.float8_e4m3fn)

    packed_w1_fp8 = convert_weight_packed(w1_fp8)
    packed_w2_fp8 = convert_weight_packed(w2_fp8)

    w1_ss = torch.randn(E, 2 * N // BLOCK_N, K // BLOCK_K)
    w2_ss = torch.randn(E, K // BLOCK_N, N // BLOCK_K)

    niters = 100
    L = 20

    inputs = [input.clone() for _ in range(L)]
    packed_w1s = [packed_w1.clone() for _ in range(L)]
    packed_w2s = [packed_w2.clone() for _ in range(L)]
    packed_w1s_int8 = [packed_w1_int8.clone() for _ in range(L)]
    packed_w2s_int8 = [packed_w2_int8.clone() for _ in range(L)]
    packed_w1s_fp8 = [packed_w1_fp8.clone() for _ in range(L)]
    packed_w2s_fp8 = [packed_w2_fp8.clone() for _ in range(L)]

    t1 = time()
    for _ in range(niters):
        for idx in range(L):
            fused_output = fused_experts(
                inputs[idx],
                packed_w1s[idx],
                packed_w2s[idx],
                topk_weight,
                topk_ids,
                inplace,
                False,
                False,
                None,
                None,
                None,
                None,
                None,
                prepack)
    t2 = time()
    tt0 = (t2 - t1) / niters * 1000 * 1000 / L # us

    t3 = time()
    for _ in range(niters):
        for idx in range(L):
            fused_output = fused_experts(
                inputs[idx],
                packed_w1s_int8[idx],
                packed_w2s_int8[idx],
                topk_weight,
                topk_ids,
                inplace,
                True,
                False,
                w1_s,
                w2_s,
                None,
                None,
                None,
                prepack)
    t4 = time()
    tt1 = (t4 - t3) / niters * 1000 * 1000 / L # us

    t5 = time()
    for _ in range(niters):
        for idx in range(L):
            fused_output = fused_experts(
                inputs[idx],
                packed_w1s_fp8[idx],
                packed_w2s_fp8[idx],
                topk_weight,
                topk_ids,
                inplace,
                False,
                True,
                w1_ss,
                w2_ss,
                [BLOCK_N, BLOCK_K],
                None,
                None,
                prepack)
    t6 = time()
    tt2 = (t6 - t5) / niters * 1000 * 1000 / L # us

    print(f"### fused_experts: M = {M}, N = {N}, K = {K}, E = {E}, TopK = {topk}: bfloat16: {tt0:.3f} us; int8: {tt1:.3f} us; fp8: {tt2:.3f}")


run_single_test(4, 352, 7168, 256, 8)
