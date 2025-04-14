import torch
import torch.nn.functional as F
import math
from sgl_kernel.common_ops import convert_weight_packed, fp8_scaled_mm_cpu
from sgl_kernel.common_ops import shared_expert_cpu as shared_expert

from utils import compare

torch.manual_seed(1111)

def SiluAndMul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]

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

def run_single_test(m, n, k, routed_scaling_factor, dtype, prepack):
    scale_block_size_N = 64
    scale_block_size_K = 128
    assert scale_block_size_N <= n
    assert scale_block_size_K <= k

    hidden_states = torch.randn(m, k, dtype=dtype) / k
    w1_fp32 = (torch.randn(2 * n, k, dtype=torch.float32) - 0.5) * 2
    w1, w1_s, w1_dq = convert_weight(w1_fp32, [scale_block_size_N, scale_block_size_K], dtype)
    w2_fp32 = (torch.randn(k, n, dtype=torch.float32) - 0.5) * 2
    w2, w2_s, w2_dq = convert_weight(w2_fp32, [scale_block_size_N, scale_block_size_K], dtype)
    fused_output = torch.randn(m, k, dtype=dtype) / k
    hidden_states2 = hidden_states.clone()

    a = torch.matmul(hidden_states, w1_dq.transpose(0, 1))
    a2 = SiluAndMul(a)
    a3 = torch.matmul(a2, w2_dq.transpose(0, 1))
    a4 = a3 + fused_output * routed_scaling_factor

    w1 = convert_weight_packed(w1)
    w2 = convert_weight_packed(w2)

    c = shared_expert(hidden_states2, w1, w2, fused_output, routed_scaling_factor, True,
                        False, True, w1_s, w2_s, [scale_block_size_N, scale_block_size_K], None, None, True)
    compare(a4, c)

run_single_test(121, 512, 512, 16, torch.bfloat16, False)
# # run_single_test(121, 32*4, 32*2, 16, torch.bfloat16)
