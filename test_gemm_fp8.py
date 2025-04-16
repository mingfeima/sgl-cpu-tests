import math
from itertools import product
from time import time

import torch
import torch.nn as nn

from sgl_kernel.common_ops import convert_weight_packed, fp8_scaled_mm_cpu
from time import time

from utils import compare

torch.manual_seed(1111)


class Mod(nn.Module):
    def __init__(self, input_channel, output_channel, has_bias):
        super(Mod, self).__init__()
        self.linear = torch.nn.Linear(input_channel, output_channel, has_bias)

    def forward(self, x):
        return self.linear(x)


def convert_weight(weight, scale_block_size, A_dtype):
    N, K = weight.size()
    fp8_max = 448.0
    scale_block_size_N, scale_block_size_K = scale_block_size  # (128, 128)

    pad_N = (scale_block_size_N - (N % scale_block_size_N)) % scale_block_size_N
    pad_K = (scale_block_size_K - (K % scale_block_size_K)) % scale_block_size_K

    if pad_N > 0 or pad_K > 0:
        weight = torch.nn.functional.pad(weight, (0, pad_K, 0, pad_N))

    weight_blocks = weight.view(
        math.ceil(N / scale_block_size_N),
        scale_block_size_N,
        math.ceil(K / scale_block_size_K),
        scale_block_size_K
    )  # (8, 128, 8, 128)
    weight_blocks = weight_blocks.permute(0, 2, 1, 3).contiguous()  # (8, 8, 128, 128)

    # Step 2: compute per-block max abs values → scale
    abs_max = weight_blocks.abs().amax(dim=(-2, -1), keepdim=True)  # (8, 8, 1, 1)
    scales = abs_max / fp8_max
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)  # avoid division by zero

    q_fp8 = (weight_blocks / scales).to(torch.float8_e4m3fn)
    q_fp8_reshape = q_fp8.permute(0, 2, 1, 3).contiguous()

    if pad_N > 0 or pad_K > 0:
        q_fp8_reshape = q_fp8_reshape.view(N + pad_N, K + pad_K)
        q_fp8_reshape = q_fp8_reshape[:N, :K].contiguous()
    else:
        q_fp8_reshape = q_fp8_reshape.view(N, K)

    dq_weight = q_fp8.float() * scales
    dq_weight = dq_weight.permute(0, 2, 1, 3).contiguous()  # (8, 128, 8, 128)

    if pad_N > 0 or pad_K > 0:
        w_dq = dq_weight.view(N + pad_N, K + pad_K).to(A_dtype)
        w_dq = w_dq[:N, :K].contiguous()
    else:
        w_dq = dq_weight.view(N, K).to(A_dtype)

    scales = scales.view(
        math.ceil(N / scale_block_size_N),
        math.ceil(K / scale_block_size_K)
    )

    return q_fp8_reshape, scales, w_dq


def run_single_test(M, N, K, has_bias, prepack, chunk=False, bench=False):
    print(f"### gemm_fp8: M = {M}, N = {N}, K = {K}, has_bias = {has_bias}, prepack = {prepack}, chunk = {chunk}")

    scale_block_size_N = 64
    scale_block_size_K = 128
    assert scale_block_size_N <= N
    assert scale_block_size_K <= K
    A_dtype = torch.bfloat16

    model = Mod(K, N, has_bias).eval()
    if chunk:
        data = torch.randn(M, K + 6, dtype=A_dtype).narrow(1, 0, K)
    else:
        data = torch.randn(M, K, dtype=A_dtype)

    weight = model.linear.weight  # (N, K)

    if has_bias:
        bias = model.linear.bias

    fp8_weight, scales, dq_weight = convert_weight(weight, [scale_block_size_N, scale_block_size_K], A_dtype)

    if has_bias:
        ref = torch.matmul(data.to(A_dtype), dq_weight.T) + bias.to(A_dtype)
    else:
        ref = torch.matmul(data.to(A_dtype), dq_weight.T)

    if prepack:
        fp8_weight = convert_weight_packed(fp8_weight)

    opt = fp8_scaled_mm_cpu(
        data,
        fp8_weight,
        scales,
        [scale_block_size_N, scale_block_size_K],
        bias if has_bias else None,
        data.dtype,
        prepack
    )
    compare(ref, opt)
    
    if bench:
        niters = 1000
        
        t0 = time()
        for _ in range(niters):
            if has_bias:
                ref = torch.matmul(data.to(A_dtype), dq_weight.T) + bias.to(A_dtype)
            else:
                ref = torch.matmul(data.to(A_dtype), dq_weight.T)
        t1 = time()
        t_native = (t1 - t0) / niters * 1000 * 1000 # us
        
        t2 = time()
        for _ in range(niters):
            fp8_scaled_mm_cpu(
                data,
                fp8_weight,
                scales,
                [scale_block_size_N, scale_block_size_K],
                bias if has_bias else None,
                data.dtype,
                prepack
            )
        t3 = time()
        t_opt = (t3 - t2) / niters * 1000 * 1000 # us
        print(f"\n### gemm_fp8 benchmark: M = {M}, N = {N}, K = {K}, has_bias = {has_bias}, prepack = {prepack}, chunk = {chunk}")
        print(f"gemm_bf16(native): {t_native:.3f} us, gemm_fp8(opt): {t_opt:.3f} us")


for M, N, K, has_bias, prepack in product([1, 2, 11, 111], [128, 224], [512, 576], [False, True], [False, True]):
    run_single_test(M, N, K, has_bias, prepack)

# test mat1_strideM
run_single_test(M=14, N=160, K=768, has_bias=True, prepack=True, chunk=True)
run_single_test(M=1, N=2816, K=7168, has_bias=True, prepack=True, chunk=False, bench=True)
