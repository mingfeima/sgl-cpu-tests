import torch
import torch.nn.functional as F

import sgl_kernel
convert_weight_packed = torch.ops.sgl_kernel.convert_weight_packed
fused_experts = torch.ops.sgl_kernel.fused_experts_cpu

import math

from utils import compare

torch.manual_seed(1111)

#torch.set_printoptions(profile="full")

def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x = x.to(torch.float32)
    d = x.shape[-1] // 2
    out = F.silu(x[..., :d]) * x[..., d:]
    return out.to(dtype)

def per_token_quant_int8(x):
    x = x.float()
    absmax = x.abs().max(dim=-1).values
    absmax = absmax.clamp_min(1e-7).unsqueeze(-1)
    scale_x = absmax / 127
    x_q = x.mul(127 / absmax)
    x_q = torch.round(x_q).to(torch.int8)

    return x_q, scale_x


### use float32 for accumulation to reduce rounding error, emulate C++ kernel behavior
### this is different from triton test case at sglang test_int_kernel.py

def native_w8a8_per_token_matmul(A, B, As, Bs, output_dtype=torch.float32):
    """Matrix multiplication function that supports per-token input quantization and per-column weight quantization"""
    A = A.to(torch.float32)
    B = B.to(torch.float32)

    assert A.shape[-1] == B.shape[-1], "Dimension mismatch"
    assert B.ndim == 2 and B.is_contiguous(), "B must be a 2D contiguous tensor"

    # Reshape input
    M = A.numel() // A.shape[-1]
    B = B.t()  # Transpose weight matrix
    N, K = B.shape
    origin_C_shape = A.shape[:-1] + (K,)
    A = A.reshape(M, N)

    # As is per-token [M, 1], Bs is per-column [1, K]
    C = torch.matmul(A, B)  # [M, K]
    C = As * C * Bs.view(1, -1)  # Broadcast per-column scale

    return C.reshape(origin_C_shape).to(output_dtype)


def torch_w8a8_per_column_moe(a, w1, w2, w1_s, w2_s, topk_weight, topk_ids, topk):
    """This function performs fused moe with per-column int8 quantization using native torch."""

    B, D = a.shape
    # Perform per-token quantization
    a_q, a_s = per_token_quant_int8(a)
    # Repeat tokens to match topk
    a_q = a_q.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    # Also repeat the scale
    a_s = a_s.view(B, -1, 1).repeat(1, topk, 1).reshape(-1, 1)  # [B*topk, 1]

    out = torch.zeros(B * topk, w2.shape[1], dtype=torch.float32, device=a.device)

    # Calculate routing
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    # Process each expert
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            # First MLP layer: note that a_s is now per-token
            inter_out = native_w8a8_per_token_matmul(
                a_q[mask], w1[i], a_s[mask], w1_s[i], output_dtype=torch.float32
            )
            # Activation function
            act_out = silu_and_mul(inter_out)
            # Quantize activation output with per-token
            act_out_q, act_out_s = per_token_quant_int8(act_out)
            # Second MLP layer
            out[mask] = native_w8a8_per_token_matmul(
                act_out_q, w2[i], act_out_s, w2_s[i], output_dtype=torch.float32
            )
    # Apply routing weights and sum
    return (
        out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1).to(a.dtype)


def run_single_test(M, N, K, E, topk, dtype, prepack):

    # Initialize int8 quantization parameters
    factor_for_scale = 1e-2
    int8_max = 127
    int8_min = -128

    # Input tensor
    # M * K
    a = torch.randn((M, K), dtype=dtype) / math.sqrt(K)

    # Generate int8 weights
    w1_fp32 = (torch.rand((E, 2 * N, K), dtype=torch.float32) - 0.5) * 2
    w1 = (w1_fp32 * int8_max).clamp(min=int8_min, max=int8_max).to(torch.int8)

    w2_fp32 = (torch.rand((E, K, N), dtype=torch.float32) - 0.5) * 2
    w2 = (w2_fp32 * int8_max).clamp(min=int8_min, max=int8_max).to(torch.int8)

    # Generate scale for each column (per-column quantization)
    w1_s = torch.rand(E, 2 * N, device=w1_fp32.device) * factor_for_scale
    w2_s = torch.rand(E, K, device=w2_fp32.device) * factor_for_scale

    # Calculate routing
    score = torch.randn((M, E), dtype=dtype)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)

    ref_out = torch_w8a8_per_column_moe(a, w1, w2, w1_s, w2_s, topk_weight, topk_ids, topk)

    inplace = True
    packed_w1 = convert_weight_packed(w1) if prepack else w1
    packed_w2 = convert_weight_packed(w2) if prepack else w2
    out = fused_experts(a, packed_w1, packed_w2, topk_weight, topk_ids.to(torch.int32), inplace, True, False, w1_s, w2_s, None, None, None, prepack)

    print("### using default atol=rtol=0.01 for torch.bfloat16: (may fail for large input shape")
    compare(ref_out, out)

    ### test_int8_kernel.py use 0.05, we use 0.01
    print("\n### same method with test_int8_kernel.py:")
    res = torch.mean(torch.abs(out.to(torch.float32) - ref_out.to(torch.float32))) / torch.mean(torch.abs(ref_out.to(torch.float32))) < 0.01
    print(res)

run_single_test(1, 128, 256, 8, 2, torch.bfloat16, prepack=False)
run_single_test(39, 1280, 256 * 4, 8, 3, torch.bfloat16, prepack=True)
run_single_test(1024, 1280, 256 * 4, 8, 3, torch.bfloat16, prepack=True)

