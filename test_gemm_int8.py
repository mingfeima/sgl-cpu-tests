import torch
from sgl_kernel.common_ops import per_token_quant_int8_cpu
from sgl_kernel.common_ops import int8_scaled_mm_cpu
from sgl_kernel.common_ops import int8_scaled_mm_with_quant
from sgl_kernel.common_ops import convert_weight_packed

from utils import compare

torch.manual_seed(1111)

#torch.set_printoptions(profile="full")

def per_token_quant_int8(x):
    x = x.float()
    absmax = x.abs().max(dim=-1).values
    absmax = absmax.clamp_min(1e-10).unsqueeze(-1)
    scale_x = absmax / 127
    x_q = x.mul(127 / absmax)
    x_q = torch.round(x_q).to(torch.int8)

    return x_q, scale_x


def native_w8a8_per_token_matmul(A, B, As, Bs, bias, output_dtype=torch.bfloat16):
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

    if bias is not None:
        C.add_(bias.view(1, -1))

    return C.reshape(origin_C_shape).to(output_dtype)


def run_single_test(M, N, K, dtype, has_bias=False):

    A = torch.randn((M, K), dtype=dtype) / 10
    Aq, As = per_token_quant_int8(A)

    factor_for_scale = 1e-2
    int8_max = 127
    int8_min = -128

    B = (torch.rand((N, K), dtype=torch.float32) - 0.5) * 2
    Bq = (B * int8_max).clamp(min=int8_min, max=int8_max).to(torch.int8)
    Bs = torch.rand(N) * factor_for_scale

    bias = torch.randn(N) if has_bias else None
    ref_out = native_w8a8_per_token_matmul(Aq, Bq, As, Bs, bias, dtype)

    Aq2, As2 = per_token_quant_int8_cpu(A)
    out = int8_scaled_mm_cpu(Aq2, Bq, As2, Bs, bias if has_bias else None, torch.bfloat16, False);

    compare(ref_out, out)

    # test the fused version
    fused_out = int8_scaled_mm_with_quant(A, Bq, Bs, bias if has_bias else None, torch.bfloat16, False);
    compare(ref_out, fused_out)


for bias in [True, False]:
    run_single_test(128, 32 * 12, 32 * 17, torch.bfloat16, bias)
    run_single_test(2, 32, 32, torch.bfloat16, bias)

def test_weight_prepack(oc, ic):

    BLOCK_N = 32

    off = torch.randint(low=-128, high=128, size=(1, ic), dtype=torch.int32).fill_(128)
    w = torch.randint(low=-128, high=128, size=(oc, ic), dtype=torch.int8)
    packed_w = convert_weight_packed(w)

    comp = torch.matmul(off, w.t().to(torch.int32))
    packed_w1 = packed_w.view(-1).narrow(0, 0, oc * ic).view(oc, ic)

    ref = w.view(int(oc/BLOCK_N), BLOCK_N, int(ic/4), 4).permute(0, 2, 1, 3).contiguous().view(oc, ic)

    print("\n### test_weight_prepack: ", torch.equal(ref, packed_w1))

#test_weight_prepack(32, 32)
