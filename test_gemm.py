import torch
import sgl_kernel

from time import time

from utils import compare

torch.manual_seed(1111)

def run_single_test(M, N, K, has_bias):

    mat1 = torch.randn(M, K, dtype=torch.bfloat16)
    mat2 = torch.randn(N, K, dtype=torch.bfloat16)

    ref = torch.matmul(mat1.float(), mat2.float().t())
    if has_bias:
        bias = torch.randn(N, dtype=torch.float32)
        ref.add_(bias.bfloat16())

    ref = ref.bfloat16()

    out = torch.ops.sgl_kernel.weight_packed_linear(mat1, mat2, bias if has_bias else None, False)

    packed_mat2 = torch.ops.sgl_kernel.convert_weight_packed(mat2)
    out2 = torch.ops.sgl_kernel.weight_packed_linear(mat1, packed_mat2, bias if has_bias else None, True)

    compare(ref, out)
    compare(ref, out2)

for has_bias in [True]:
    run_single_test(1, 32 * 13, 32 * 16, has_bias)
    run_single_test(101, 32 * 13, 32 * 16, has_bias)


def test_weight_prepack(oc, ic):

    BLOCK_N = 32

    w1 = torch.randn(oc, ic, dtype = torch.bfloat16)
    packed_w1 = torch.ops.sgl_kernel.convert_weight_packed(w1)
    ref = w1.view(int(oc/BLOCK_N), BLOCK_N, int(ic/2), 2).permute(0, 2, 1, 3).contiguous().view(oc, ic)

    print("\n### test_weight_prepack: ", torch.equal(ref, packed_w1))

test_weight_prepack(16 * 8, 32 * 24)
#test_weight_prepack(160, 3072, 5120)

def run_single_bench(M, N, K):

    mat1 = torch.randn(M, K, dtype=torch.bfloat16)
    mat2 = torch.randn(N, K, dtype=torch.bfloat16)
    packed_mat2 = torch.ops.sgl_kernel.convert_weight_packed(mat2)

    L = 100
    mat1 = [torch.randn(M, K, dtype=torch.bfloat16) for _ in range(L)]
    mat2 = [torch.randn(N, K, dtype=torch.bfloat16) for _ in range(L)]
    packed_mat2 = [torch.ops.sgl_kernel.convert_weight_packed(t) for t in mat2]

    niters = 500
    for _ in range(int(niters / 100)):
        for l in range(L):
            out = torch.ops.sgl_kernel.weight_packed_linear(mat1[l], packed_mat2[l], None, True)

    t1 = time()
    for _ in range(niters):
        for l in range(L):
            out = torch.ops.sgl_kernel.weight_packed_linear(mat1[l], packed_mat2[l], None, True)
    t2 = time()
    tt = (t2 - t1) / niters / L * 1000 * 1000 # us

    print(f"\n### bench: gemm: M = {M}, N = {N}, K = {K}, time = {tt:.3f} us")

run_single_bench(1, 256, 7168)
