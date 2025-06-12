import math
from itertools import product
from time import time
import math

import torch
import torch.nn as nn

from sgl_kernel.common_ops import convert_weight_packed, fp8_scaled_mm_cpu
from time import time

from utils import compare

torch.manual_seed(1111)

BLOCK_N, BLOCK_K = 64, 128
factor_for_scale = 1e-3
fp8_max, fp8_min = 400, -400

def scaled_weight(weight, scales):
    N, K = weight.shape
    weight_block = weight.view(N // BLOCK_N, BLOCK_N, K // BLOCK_K, BLOCK_K).permute(0, 2, 1, 3).float().contiguous()
    return (weight_block * scales.view(N // BLOCK_N, K // BLOCK_K, 1, 1)).permute(0, 2, 1, 3).contiguous().view(N, K)

def run_single_test(M, N, K, has_bias, prepack, chunk=False, bench=False):
    print(f"\n### gemm_fp8: M = {M}, N = {N}, K = {K}, has_bias = {has_bias}, prepack = {prepack}, chunk = {chunk}")

    A_dtype = torch.bfloat16

    if chunk:
        data = torch.randn(M, K + 6, dtype=A_dtype).narrow(1, 0, K)
    else:
        data = torch.randn(M, K, dtype=A_dtype)
    data = data / math.sqrt(K)

    weight_fp32 = torch.randn(N, K)
    scales = torch.randn(N // BLOCK_N, K // BLOCK_K) * factor_for_scale
    weight_fp8 = (weight_fp32 * fp8_max).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    weight_scaled = scaled_weight(weight_fp8, scales).view(N, K).to(dtype=A_dtype)

    if has_bias:
        bias = torch.randn(N, dtype=torch.float32)

    if has_bias:
        ref = torch.matmul(data.to(A_dtype), weight_scaled.T) + bias.to(A_dtype)
    else:
        ref = torch.matmul(data.to(A_dtype), weight_scaled.T)

    if prepack:
        weight_fp8 = convert_weight_packed(weight_fp8)

    opt = fp8_scaled_mm_cpu(
        data,
        weight_fp8,
        scales,
        [BLOCK_N, BLOCK_K],
        bias if has_bias else None,
        data.dtype,
        prepack
    )
    compare(ref, opt)
    
    if bench:
        niters = 200
        L = 100

        inputs = [data.clone() for _ in range(L)]
        weights_fp8 = [weight_fp8.clone() for _ in range(L)]
        
        t0 = time()
        for _ in range(niters):
            if has_bias:
                ref = torch.matmul(data.to(A_dtype), weight_scaled.T) + bias.to(A_dtype)
            else:
                ref = torch.matmul(data.to(A_dtype), weight_scaled.T)
        t1 = time()
        t_native = (t1 - t0) / niters * 1000 * 1000 # us
 
        for _ in range(niters):
            for idx in range(L):
                fp8_scaled_mm_cpu(
                    inputs[idx],
                    weights_fp8[idx],
                    scales,
                    [BLOCK_N, BLOCK_K],
                    bias if has_bias else None,
                    data.dtype,
                    prepack
                )
        
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
                    prepack
                )
        t3 = time()
        t_opt = (t3 - t2) / niters * 1000 * 1000 / L # us
        print(f"\n### gemm_fp8 benchmark: M = {M}, N = {N}, K = {K}, has_bias = {has_bias}, prepack = {prepack}, chunk = {chunk}")
        print(f"gemm_bf16(native): {t_native:.3f} us, gemm_fp8(opt): {t_opt:.3f} us")


#for M, N, K, has_bias, prepack in product([1, 2, 11, 111], [128, 224], [512, 576], [False, True], [False, True]):
#    run_single_test(M, N, K, has_bias, prepack)

# test mat1_strideM
#run_single_test(M=14, N=160, K=768, has_bias=True, prepack=True, chunk=True)
#run_single_test(M=1, N=2816, K=7168, has_bias=True, prepack=True, chunk=False, bench=True)

#run_single_test(1, 576, 7168, True, True, False, True)
run_single_test(1, 2816, 7168, True, True, False, True)
run_single_test(2, 2816, 7168, True, True, False, True)
run_single_test(4, 2816, 7168, True, True, False, False)
run_single_test(1024, 2816, 7168, True, True, False, False)
#run_single_test(128, 2816, 7168, True, True, False, True)
