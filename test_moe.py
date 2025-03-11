import torch
import torch.nn.functional as F
from sgl_kernel.ops._kernels import fused_experts_cpu as fused_experts
from sgl_kernel.ops._kernels import grouped_topk_cpu as grouped_topk
from sgl_kernel.ops._kernels import convert_weight_packed

from utils import compare

torch.manual_seed(1111)

NUM_EXPERTS = [8, 64]
TOP_KS = [2, 6]

def SiluAndMul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def torch_naive_moe(a, w1, w2, score, topk, renormalize):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)

    if renormalize:
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)

    #print("ref_topk_weight: \n", topk_weight, topk_weight.size())
    #print("ref_topk_ids   : \n", topk_ids, topk_ids.size())

    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            # debug print
            #res = a[mask] @ w1[i].transpose(0, 1)
            #res0 = res.narrow(1, 0, 128)
            #res1 = res.narrow(1, 128, 128)
            #print("### expert = ", i, "; a @ w1 top half =\n", res0, res0.size())
            #print("### expert = ", i, "; a @ w1 bottom half =\n", res1, res1.size())
            #ic1 = SiluAndMul(a[mask] @ w1[i].transpose(0, 1))
            #print("### expert = ", i, ";  ic1 = ", ic1.size(), "\n", ic1)
            out[mask] = SiluAndMul(a[mask] @ w1[i].transpose(0, 1)) @ w2[
                i
            ].transpose(0, 1)
            #print("## out, ic2 = ", out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1))
    return (
        out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)


def fused_moe(a, w1, w2, score, topk, renormalize, prepack):

    G = 1
    topk_group = 1

    B, D = a.shape
    topk_weights = torch.empty(B, topk, dtype=torch.float32)
    topk_ids = torch.empty(B, topk, dtype=torch.int32)
    grouped_topk(
        topk_weights,
        topk_ids,
        a,
        score,
        topk,
        renormalize,
        G,
        topk_group)

    #print(topk_weights, topk_weights.size())
    #print(topk_ids, topk_ids.size())

    packed_w1 = convert_weight_packed(w1) if prepack else w1
    packed_w2 = convert_weight_packed(w2) if prepack else w2

    inplace = True
    return fused_experts(a, packed_w1, packed_w2, topk_weights, topk_ids, inplace, prepack)


def run_single_test(m, n, k, e, topk, dtype, renormalize=False, use_fp8_w8a8=False, prepack=False):

    a = torch.randn((m, k), device="cpu", dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device="cpu", dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device="cpu", dtype=dtype) / 10
    score = torch.randn((m, e), device="cpu", dtype=dtype)

    torch_output = torch_naive_moe(a, w1, w2, score, topk, renormalize)
    fused_output = fused_moe(a, w1, w2, score, topk, renormalize, prepack)
    
    #print("torch_output: ", torch_output, torch_output.size())
    #print("fused_output: ", fused_output, fused_output.size())
    res = compare(torch_output, fused_output)


run_single_test(2, 32, 32, 4, 2, torch.bfloat16)
run_single_test(2, 128, 32, 4, 2, torch.bfloat16, renormalize=True, prepack=False)
run_single_test(2, 128, 32, 4, 2, torch.bfloat16, renormalize=True, prepack=True)
run_single_test(114, 4096, 1024 + 32, 8, 2, torch.bfloat16, renormalize=True)


def test_weight_prepack(e, oc, ic):

    BLOCK_N = 32

    w1 = torch.randn(e, oc, ic, dtype = torch.bfloat16)
    packed_w1 = convert_weight_packed(w1)
    ref = w1.view(e, int(oc/BLOCK_N), BLOCK_N, int(ic/2), 2).permute(0, 1, 3, 2, 4).contiguous().view(e, oc, ic)

    print("\n### test_weight_prepack: ", torch.equal(ref, packed_w1))

test_weight_prepack(256, 16 * 8, 32 * 24)
