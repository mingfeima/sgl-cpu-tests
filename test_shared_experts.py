import torch
import torch.nn.functional as F
#from test_moe_int8 import native_w8a8_per_token_matmul
from sgl_kernel.common_ops import convert_weight_packed
from sgl_kernel.common_ops import shared_expert_cpu as shared_expert

from utils import compare

torch.manual_seed(1111)

def SiluAndMul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]

def per_token_quant_int8(x):
    x = x.float()
    absmax = x.abs().max(dim=-1).values
    absmax = absmax.clamp_min(1e-7).unsqueeze(-1)
    scale_x = absmax / 127
    x_q = x.mul(127 / absmax)
    x_q = torch.round(x_q).to(torch.int8)

    return x_q, scale_x

def native_w8a8_per_token_matmul(A, B, As, Bs, output_dtype=torch.float32):
    A = A.to(torch.float32)
    B = B.to(torch.float32)

    C = torch.matmul(A, B.t())  # [M, K]
    C = As* C * Bs.view(1, -1)  # Broadcast per-column scale

    return C.to(output_dtype)

def torch_naive_moe(a, w1, w2, b, routed_scaling_factor):

    ic1 = torch.matmul(a, w1.transpose(0, 1))
    ic2 = SiluAndMul(ic1)
    ic3 = torch.matmul(ic2, w2.transpose(0, 1))

    return ic3 + b * routed_scaling_factor

def torch_w8a8_per_column_moe(a, w1_q, w2_q, w1_s, w2_s, b, routed_scaling_factor):

    # Perform per-token quantization
    a_q, a_s = per_token_quant_int8(a)

    ic1 = native_w8a8_per_token_matmul(a_q, w1_q, a_s, w1_s)
    ic2 = SiluAndMul(ic1)

    a1_q, a1_s = per_token_quant_int8(ic2)
    ic3 = native_w8a8_per_token_matmul(a1_q, w2_q, a1_s, w2_s)

    return ic3 + b * routed_scaling_factor


def run_single_test(m, n, k, routed_scaling_factor, dtype, prepack=False):

    hidden_states = torch.randn(m, k, dtype=dtype) / k
    w1 = torch.randn(2 * n, k, dtype=dtype)
    w2 = torch.randn(k, n, dtype=dtype)
    fused_output = torch.randn(m, k, dtype=dtype) / k

    # fused moe mutates content in hs
    hidden_states2 = hidden_states.clone()

    # bfloat16
    ref = torch_naive_moe(hidden_states.float(), w1.float(), w2.float(), fused_output.float(), routed_scaling_factor).to(dtype=dtype)
    res = shared_expert(hidden_states, w1, w2, fused_output, routed_scaling_factor, True, False, None, None, None, None, False)

    #print(ref, ref.size())
    #print(res, res.size())
    compare(ref, res)

    # int8
    w1_q, w1_s = per_token_quant_int8(w1)
    w2_q, w2_s = per_token_quant_int8(w2)
    ref2 = torch_w8a8_per_column_moe(hidden_states2.float(), w1_q, w2_q, w1_s, w2_s, fused_output.float(), routed_scaling_factor).to(dtype=dtype)
    res2 = shared_expert(hidden_states2, w1_q, w2_q, fused_output, routed_scaling_factor, True, True, w1_s, w2_s, None, None, False)

    #print(ref2, ref2.size())
    #print(res2, res2.size())
    compare(ref2, res2)


run_single_test(2, 32, 32, 16, torch.bfloat16)
run_single_test(121, 32*4, 32*2, 16, torch.bfloat16)
