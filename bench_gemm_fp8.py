import math
from itertools import product
from time import time
import math

import torch
import torch.nn as nn

from sgl_kernel.common_ops import convert_weight_packed, fp8_scaled_mm_cpu
from sgl_kernel.common_ops import int8_scaled_mm_with_quant, convert_weight_packed, weight_packed_linear
from time import time

from utils import compare

torch.manual_seed(1111)

BLOCK_N, BLOCK_K = 128, 128
factor_for_scale = 1e-3
fp8_max, fp8_min = 400, -400

def per_token_quant_int8(x):
    x = x.float()
    absmax = x.abs().max(dim=-1).values
    absmax = absmax.clamp_min(1e-10).unsqueeze(-1)
    scale_x = absmax / 127
    x_q = x.mul(127 / absmax)
    x_q = torch.round(x_q).to(torch.int8)

    return x_q, scale_x

def scaled_weight(weight, scales):
    N, K = weight.shape
    weight_block = weight.view(N // BLOCK_N, BLOCK_N, K // BLOCK_K, BLOCK_K).permute(0, 2, 1, 3).float().contiguous()
    return (weight_block * scales.view(N // BLOCK_N, K // BLOCK_K, 1, 1)).permute(0, 2, 1, 3).contiguous().view(N, K)

def run_single_test(M, N, K, has_bias):
    A_dtype = torch.bfloat16

    data = torch.randn(M, K, dtype=A_dtype)
    data = data / math.sqrt(K)

    # fp8 weights and scales
    weight_fp32 = torch.randn(N, K)
    scales = torch.randn(N // BLOCK_N, K // BLOCK_K) * factor_for_scale
    weight_fp8 = (weight_fp32 * fp8_max).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    weight_scaled = scaled_weight(weight_fp8, scales).view(N, K).to(dtype=A_dtype)
    weight_fp8 = convert_weight_packed(weight_fp8)

    # int8 weights and scales
    int8_max = 127
    int8_min = -128
    B = (torch.rand((N, K), dtype=torch.float32) - 0.5) * 2
    Bq = (B * int8_max).clamp(min=int8_min, max=int8_max).to(torch.int8)
    Bs = torch.rand(N) * factor_for_scale
    weight_int8 = convert_weight_packed(Bq)

    if has_bias:
        bias = torch.randn(N, dtype=torch.float32)

    niters = 200
    L = 300

    inputs = [data.clone() for _ in range(L)]
    weight_bf16 = torch.randn(N, K, dtype=torch.bfloat16)
    weights_bf16 = [weight_bf16.clone() for _ in range(L)]
    weights_fp8 = [weight_fp8.clone() for _ in range(L)]
    weights_int8 = [weight_int8.clone() for _ in range(L)]
        
    t0 = time()
    for _ in range(niters):
        for idx in range(L):
            ref = torch.matmul(inputs[idx], weights_bf16[idx].T)
    t1 = time()
    tt0 = (t1 - t0) / niters * 1000 * 1000 / L # us

    t2 = time()
    for _ in range(niters):
        for idx in range(L):
            fp8_scaled_mm_cpu(
                inputs[idx],
                weights_fp8[idx],
                scales,
                [BLOCK_N, BLOCK_K],
                bias if has_bias else None,
                data.dtype,
                True
            )
    t3 = time()
    tt1 = (t3 - t2) / niters * 1000 * 1000 / L # us

    t4 = time()
    for _ in range(niters):
        for idx in range(L):
            int8_scaled_mm_with_quant(
                inputs[idx],
                weights_int8[idx],
                Bs,
                bias if has_bias else None,
                A_dtype,
                True);
    t5 = time()
    tt2 = (t5 - t4) / niters * 1000 * 1000 / L # us

    t6 = time()
    for _ in range(niters):
        for idx in range(L):
            weight_packed_linear(
                inputs[idx],
                weights_bf16[idx],
                bias if has_bias else None,
                True)
    t7 = time()
    tt3 = (t7 - t6) / niters * 1000 * 1000 / L # us
    
    print(f"\n### gemm_fp8 benchmark: M = {M}, N = {N}, K = {K}, has_bias = {has_bias}")
    print(f"gemm_bf16(native): {tt0:.3f} us, gemm_fp8(opt): {tt1:.3f} us, gemm_int8(opt): {tt2:.3f} us, gemm_bf16(opt): {tt3:.3f} us")


run_single_test(4, 2816, 7168, False)
#run_single_test(128, 2816, 7168, True)
