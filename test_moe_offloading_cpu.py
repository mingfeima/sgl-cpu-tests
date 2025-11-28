import torch
import torch.nn.functional as F
import math
import sgl_kernel

fused_experts = torch.ops.sgl_kernel.fused_experts_cpu
convert_weight_packed = torch.ops.sgl_kernel.convert_weight_packed

torch.manual_seed(1111)

# test case configs
num_topk = 8
num_experts = 128
num_experts_offload_cpu = 8
expert_padding_id = -1


# fp8 block quantize
BLOCK_N, BLOCK_K = 128, 128
factor_for_scale = 1e-3
fp8_max, fp8_min = 400, -400

def SiluAndMul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]

# apply block scale to weights
def scaled_weight(weight, scales):
    E, N, K = weight.shape
    weight_block = weight.view(E, N // BLOCK_N, BLOCK_N, K // BLOCK_K, BLOCK_K).permute(0, 1, 3, 2, 4).float().contiguous()
    return (weight_block * scales.view(E, N // BLOCK_N, K // BLOCK_K, 1, 1)).permute(0, 1, 3, 2, 4).contiguous().view(E, N, K)

def native_fused_moe(a, w1, w2, topk_weight, topk_ids, topk):
    B, D = a.shape
    old_dtype = a.dtype
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D).float()
    out = torch.zeros(B * topk, w2.shape[1], dtype=torch.float32)

    # Calculate routing
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)

    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            ic0 = torch.matmul(a[mask], w1[i].transpose(0, 1))
            ic1 = SiluAndMul(ic0)
            out[mask] = torch.matmul(ic1, w2[i].transpose(0, 1))

    return (
        out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1).to(old_dtype)

def test_fused_expert(M, N, K, E, topk, dtype, prepack=False):
    print(f"\n### test_fused_expert: M = {M}, N = {N}, K = {K}, E = {E}, topk = {topk}")

    topk_ids = torch.randint(0, E, (M, topk), dtype=torch.int32)

    ratio = num_topk * 1. / num_experts
    print("### setting ratio to be topk / num_experts = {:.5f}".format(ratio))

    # mask `topk_ids` with ratio, pad with -1
    mask = torch.rand_like(topk_ids.float()) < ratio
    topk_ids[~mask] = expert_padding_id

    # remove rows that are all -1
    mask = ~(topk_ids == expert_padding_id).all(dim=1)
    topk_ids = topk_ids[mask]

    # the actual batch size for CPU, after removing tokens not selected
    actual_M = topk_ids.size(0)
    assert(actual_M > 0)
    print("### actual_M: ", actual_M)

    print(topk_ids)
    topk_weight = torch.randn(actual_M, topk)

    a = torch.randn(actual_M, K, dtype=dtype) / math.sqrt(K)

    w1_fp32 = torch.randn(E, 2 * N, K)
    w1 = (w1_fp32 * fp8_max).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    w2_fp32 = torch.randn(E, K, N)
    w2 = (w2_fp32 * fp8_max).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    w1s = torch.randn(E, 2 * N // BLOCK_N, K // BLOCK_K) * factor_for_scale
    w2s = torch.randn(E, K // BLOCK_N, N // BLOCK_K) * factor_for_scale

    w1_scaled = scaled_weight(w1, w1s)
    w2_scaled = scaled_weight(w2, w2s)

    w1 = convert_weight_packed(w1)
    w2 = convert_weight_packed(w2)

    ref_out = native_fused_moe(
        a,
        w1_scaled,
        w2_scaled,
        topk_weight,
        topk_ids,
        topk)

    out = fused_experts(
        a,
        w1,
        w2,
        topk_weight,
        topk_ids.to(torch.int32),
        False,
        False,
        True,
        w1s,
        w2s,
        [BLOCK_N, BLOCK_K],
        None,
        None,
        True)

    res = torch.allclose(ref_out.bfloat16(), out, atol=1e-2, rtol=1e-2)
    print("### test fused experts on float8 moe, result: ", res)

    # bfloat16
    out2 = fused_experts(
        a,
        w1_scaled.bfloat16(),
        w2_scaled.bfloat16(),
        topk_weight,
        topk_ids.to(torch.int32),
        False,
        False,
        False,
        None,
        None,
        None,
        None,
        None,
        False)

    #print(ref_out.bfloat16())
    #print(out2)
    res2 = torch.allclose(ref_out.bfloat16(), out2, atol=1e-2, rtol=1e-2)
    print("### test fused experts on bfloat16 moe, result: ", res2)


test_fused_expert(14, 128, 128, 8, num_topk, torch.bfloat16)
