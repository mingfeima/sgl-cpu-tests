import torch
import torch.nn.functional as F
import sgl_kernel
from time import time

from utils import compare

torch.manual_seed(1111)

# from flashinfer
def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()

# from sglang
def input_to_float8(x, dtype = torch.float8_e4m3fn):
    """This function quantizes input values to float8 values with tensor-wise quantization."""
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    fp8_max = finfo.max
    scale = fp8_max / amax
    x_scl_sat = (x * scale).clamp(min=-fp8_max, max=fp8_max)
    return x_scl_sat.to(dtype).contiguous(), scale.float().reciprocal()

def native_bmm_fp8(matA, matB, s, matC):
    bmm_out = torch.bmm(matA, matB.to(torch.bfloat16)) * s
    matC = bmm_out

    return matC


def bmm_opt(matC, matA, matB, is_vnni, scale=None):
    torch.ops.sgl_kernel.bmm_cpu(matC, matA, matB, is_vnni, scale)

def run_single_test(B, M, N, K, chunk=False, dtype=torch.bfloat16):

    print("### bmm_fp8: B = {}, M = {}, N = {}, K = {}, chunk = {}".format(B, M, N, K, chunk))
    if chunk:
        matA = torch.randn(M, B, K + 64, dtype=dtype).narrow(2, 0, K).transpose_(0, 1)
        matB = torch.randn(B, N, K, dtype=dtype).transpose_(1, 2)
        matC = torch.randn(M, B, N + 64, dtype=dtype).narrow(2, 0, N).transpose_(0, 1)
    else:
        matA = torch.randn(M, B, K, dtype=dtype).transpose_(0, 1)
        matB = torch.randn(B, N, K, dtype=dtype).transpose_(1, 2)
        matC = torch.randn(M, B, N, dtype=dtype).transpose_(0, 1)

    #print("### matA: ", matA.size(), matA.stride())
    #print("### matB: ", matB.size(), matB.stride())
    #print("### matC: ", matC.size(), matC.stride())

    matB_q, matB_s = input_to_float8(matB)

    ref = torch.bmm(matA, matB)

    matB_t = matB.transpose_(1, 2)
    assert matB_t.is_contiguous()

    matC.zero_()
    assert matC.sum().item() == 0
    bmm_opt(matC, matA, matB_t, False)
    compare(ref, matC)

    packed_B = torch.ops.sgl_kernel.convert_weight_packed(matB_t)
    matC.zero_()
    assert matC.sum().item() == 0
    bmm_opt(matC, matA, packed_B, True)
    compare(ref, matC)


def run_single_bench(B, M, N, K, chunk=False, dtype=torch.bfloat16):

    if chunk:
        matA = torch.randn(M, B, K + 64, dtype=dtype).narrow(2, 0, K).transpose_(0, 1)
        matB = torch.randn(B, N, K, dtype=dtype).transpose_(1, 2)
        matC = torch.randn(M, B, N + 64, dtype=dtype).narrow(2, 0, N).transpose_(0, 1)
    else:
        matA = torch.randn(M, B, K, dtype=dtype).transpose_(0, 1)
        matB = torch.randn(B, N, K, dtype=dtype).transpose_(1, 2)
        matC = torch.randn(M, B, N, dtype=dtype).transpose_(0, 1)

    matB_q, matB_s = input_to_float8(matB)

    profile = False
    niters = 5000
    with torch.autograd.profiler.profile(enabled=profile) as prof:
        t1 = time()
        for _ in range(niters):
            matC = torch.bmm(matA, matB)
        t2 = time()
    if profile:
        print(prof.key_averages().table(sort_by="cpu_time_total"))

    t3 = time()
    for _ in range(niters):
        native_bmm_fp8(matA, matB_q, matB_s, matC)
    t4 = time()

    tt1 = (t2 - t1) / niters * 1000 * 1000 # us
    tt2 = (t4 - t3) / niters * 1000 * 1000 # us

    matB_t = matB.transpose_(1, 2)
    assert matB_t.is_contiguous()

    packed_B = torch.ops.sgl_kernel.convert_weight_packed(matB_t)
    t5 = time()
    for _ in range(niters):
        bmm_opt(matC, matA, packed_B, True)
    t6 = time()

    tt3 = (t6 - t5) / niters * 1000 * 1000

    print("### bmm benchmark: B = {}, M = {}, N = {}, K = {}; bmm(bf16): {:.3f} us, bmm(native fp8): {:.3f} us, bmm (opt bf16): {:.3f} us".format(B, M, N, K, tt1, tt2, tt3))


for m in [1, 2, 11, 111]:
    run_single_test(17, m, 128 + 32, 512 + 32, chunk=True)
    run_single_test(16, m, 512, 128 + 32, chunk=True)

run_single_test(1, 5, 64 + 32, 128 + 32, chunk=True)

run_single_bench(16, 1, 512, 128, chunk=True)
run_single_bench(16, 1, 128, 512, chunk=True)
#run_single_bench(16, 4, 512, 128, chunk=True)
#run_single_bench(16, 4, 128, 512, chunk=True)
#run_single_bench(16, 8, 512, 128, chunk=True)
#run_single_bench(16, 8, 128, 512, chunk=True)
#run_single_bench(16, 12, 512, 128, chunk=True)
#run_single_bench(16, 12, 128, 512, chunk=True)
#run_single_bench(16, 16, 512, 128, chunk=True)
#run_single_bench(16, 16, 128, 512, chunk=True)
#run_single_bench(16, 32, 512, 128, chunk=True)
#run_single_bench(16, 32, 128, 512, chunk=True)
#run_single_bench(16, 64, 512, 128, chunk=True)
#run_single_bench(16, 64, 128, 512, chunk=True)
#run_single_bench(16, 160, 512, 128, chunk=True)
#run_single_bench(16, 160, 128, 512, chunk=True)
#run_single_bench(8, 1, 1536, 576, chunk=True)
#run_single_bench(8, 1, 1280, 2048, chunk=True)
