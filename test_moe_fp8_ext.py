import torch
import torch.nn.functional as F
import math
import sgl_kernel
convert_weight_packed = torch.ops.sgl_kernel.convert_weight_packed
fp8_scaled_mm = torch.ops.sgl_kernel.fp8_scaled_mm_cpu
shared_expert = torch.ops.sgl_kernel.shared_expert_cpu
fused_experts = torch.ops.sgl_kernel.fused_experts_cpu

from utils import compare

torch.manual_seed(1111)

BLOCK_N, BLOCK_K = 64, 128
factor_for_scale = 1e-3
fp8_max, fp8_min = 400, -400

def SiluAndMul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]

def scaled_weight(weight, scales):
    E, N, K = weight.shape
    weight_block = weight.view(E, N // BLOCK_N, BLOCK_N, K // BLOCK_K, BLOCK_K).permute(0, 1, 3, 2, 4).float().contiguous()
    return (weight_block * scales.view(E, N // BLOCK_N, K // BLOCK_K, 1, 1)).permute(0, 1, 3, 2, 4).contiguous().view(E, N, K)

def test_shared_expert(M, N, K, routed_scaling_factor, dtype, prepack=False):
    print(f"\ntest_shared_expert: M = {M}, N = {N}, K = {K}")
    a = torch.randn(M, K, dtype=dtype) / math.sqrt(K)

    w1_fp32 = torch.randn(1, 2 * N, K)
    w1 = (w1_fp32 * fp8_max).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    w2_fp32 = torch.randn(1, K, N)
    w2 = (w2_fp32 * fp8_max).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    w1s = torch.randn(1, 2 * N // BLOCK_N, K // BLOCK_K) * factor_for_scale
    w2s = torch.randn(1, K // BLOCK_N, N // BLOCK_K) * factor_for_scale

    w1_scaled = scaled_weight(w1, w1s).view(2 * N, K)
    w2_scaled = scaled_weight(w2, w2s).view(K, N)

    # change back to 2D
    w1, w2 = w1.squeeze(0), w2.squeeze(0)
    w1s, w2s = w1s.squeeze(0), w2s.squeeze(0)
    w1_scaled, w2_scaled = w1_scaled.squeeze(0), w2_scaled.squeeze(0)

    fused_out = torch.randn(M, K, dtype=dtype) / math.sqrt(K)
    a2 = a.clone()

    # ref
    ic0 = torch.matmul(a.float(), w1_scaled.transpose(0, 1))
    ic1 = SiluAndMul(ic0)
    shared_out = torch.matmul(ic1, w2_scaled.transpose(0, 1))
    ref_out = shared_out + fused_out.float() * routed_scaling_factor
    ref_out = ref_out.to(dtype=dtype)

    w1 = convert_weight_packed(w1) # [2N, K]
    w2 = convert_weight_packed(w2) # [K, N]
    out = shared_expert(a2, w1, w2, fused_out, routed_scaling_factor, True,
                      False, True, w1s, w2s, [BLOCK_N, BLOCK_K], None, None, True)

    compare(ref_out, out)

test_shared_expert(2, 256, 1024, 16, torch.bfloat16)
test_shared_expert(12, 128, 256, 16, torch.bfloat16)
test_shared_expert(1212, 128, 256, 16, torch.bfloat16)


def native_fused_moe(a, w1, w2, topk_weight, topk_ids, topk):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D).float()
    out = torch.zeros(B * topk, w2.shape[1], dtype=torch.float32, device=a.device)

    # Calculate routing
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)

    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            ic0 = torch.matmul(a[mask], w1[i].transpose(0, 1))
            ic1 = SiluAndMul(ic0)
            out[mask] = torch.matmul(ic1, w2[i].transpose(0, 1))
            #print("@@@ ic0: at expert ", i)
            #print(ic0)


    return (
        out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1).to(a.dtype)


def test_fused_expert(M, N, K, E, topk, dtype, prepack=False):
    print(f"\n### test_fused_expert: M = {M}, N = {N}, K = {K}, E = {E}, topk = {topk}")
    a = torch.randn(M, K, dtype=dtype) / math.sqrt(K)

    w1_fp32 = torch.randn(E, 2 * N, K)
    w1 = (w1_fp32 * fp8_max).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    w2_fp32 = torch.randn(E, K, N)
    w2 = (w2_fp32 * fp8_max).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    w1s = torch.randn(E, 2 * N // BLOCK_N, K // BLOCK_K) * factor_for_scale
    w2s = torch.randn(E, K // BLOCK_N, N // BLOCK_K) * factor_for_scale

    w1_scaled = scaled_weight(w1, w1s)
    w2_scaled = scaled_weight(w2, w2s)

    score = torch.randn((M, E), dtype=dtype)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)

    w1 = convert_weight_packed(w1)
    w2 = convert_weight_packed(w2)

    ref_out = native_fused_moe(a, w1_scaled, w2_scaled, topk_weight, topk_ids, topk)
    out = fused_experts(a, w1, w2, topk_weight, topk_ids.to(torch.int32), False, False,True, w1s, w2s, [BLOCK_N, BLOCK_K], None, None, True)

    compare(ref_out.bfloat16(), out)

test_fused_expert(2, 128, 128, 8, 4, torch.bfloat16)
test_fused_expert(121, 512, 1024, 8, 2, torch.bfloat16)
test_fused_expert(1212, 512, 1024, 8, 2, torch.bfloat16)
