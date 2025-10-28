import torch
import sgl_kernel
from time import time

fused_experts = torch.ops.sgl_kernel.fused_experts_cpu
convert_weight_packed = torch.ops.sgl_kernel.convert_weight_packed

torch.manual_seed(1111)

# test case configs
num_topk = 8
num_experts = 128
num_experts_offload_cpu = 8
expert_padding_id = -1

BLOCK_N, BLOCK_K = 128, 128

def run_single_test(M, N, K, E, topk, dtype=torch.bfloat16):

    print(f"\n### bench_fused_expert: M = {M}, N = {N}, K = {K}, E = {E}, topk = {topk}")

    topk_ids = torch.randint(0, E, (M, topk), dtype=torch.int32)

    ratio = num_topk * 1. / num_experts

    # mask `topk_ids` with ratio, pad with -1
    mask = torch.rand_like(topk_ids.float()) < ratio
    topk_ids[~mask] = expert_padding_id

    # non-zeros
    nnz = mask.sum().item()
    experts_per_token = mask.sum(dim=-1)
    max, min = experts_per_token.max().item(), experts_per_token.min().item()

    # remove rows that are all -1
    mask = ~(topk_ids == expert_padding_id).all(dim=1)
    topk_ids = topk_ids[mask]

    # the actual batch size for CPU, after removing tokens not selected
    actual_M = topk_ids.size(0)
    assert(actual_M > 0)

    topk_weight = torch.randn(actual_M, topk)

    print("### setting ratio to be topk / num_experts = {:.5f}, nnz = {}, actual_M = {}, expert count per token range from {} to {}"
        .format(ratio, nnz, actual_M, min, max))

    prepack = True
    inplace = True

    input = torch.randn(actual_M, K).to(dtype=dtype)
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

    niters = 2000
    L = 20 if actual_M < 10 else 2

    inputs = [input.clone() for _ in range(L)]
    packed_w1s = [packed_w1.clone() for _ in range(L)]
    packed_w2s = [packed_w2.clone() for _ in range(L)]
    packed_w1s_int8 = [packed_w1_int8.clone() for _ in range(L)]
    packed_w2s_int8 = [packed_w2_int8.clone() for _ in range(L)]
    packed_w1s_fp8 = [packed_w1_fp8.clone() for _ in range(L)]
    packed_w2s_fp8 = [packed_w2_fp8.clone() for _ in range(L)]

    use_int4_w4a16 = False

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
                #use_int4_w4a16,
                None,
                None,
                None,
                None,
                None,
                #None,
                #None,
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
                #use_int4_w4a16,
                w1_s,
                w2_s,
                None,
                None,
                None,
                #None,
                #None,
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
                #use_int4_w4a16,
                w1_ss,
                w2_ss,
                [BLOCK_N, BLOCK_K],
                None,
                None,
                #None,
                #None,
                prepack)
    t6 = time()
    tt2 = (t6 - t5) / niters * 1000 * 1000 / L # us

    if M > 100 or True:
        # convert to ms
        tt0 = tt0 / 1000
        tt1 = tt1 / 1000
        tt2 = tt2 / 1000
        print(f"### fused_experts: M = {M}, N = {N}, K = {K}, E = {E}, TopK = {topk}: bfloat16: {tt0:.3f} ms; int8: {tt1:.3f} ms; fp8: {tt2:.3f} ms")
    else:
        print(f"### fused_experts: M = {M}, N = {N}, K = {K}, E = {E}, TopK = {topk}: bfloat16: {tt0:.3f} us; int8: {tt1:.3f} us; fp8: {tt2:.3f} us")

# TODO: test 352
run_single_test(64, 256, 4096, num_experts_offload_cpu, num_topk)
