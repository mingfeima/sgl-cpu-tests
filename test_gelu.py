import torch
import torch.nn as nn
import torch.nn.functional as F

import sgl_kernel
from time import time

from utils import compare


convert_weight_packed = torch.ops.sgl_kernel.convert_weight_packed
weight_packed_linear = torch.ops.sgl_kernel.weight_packed_linear
fused_linear_gelu_linear = torch.ops.sgl_kernel.fused_linear_gelu_linear

torch.manual_seed(1111)


def bench_gelu():

    L = 20

    #input = torch.randn(48960, 1, 1440).bfloat16()
    input = torch.randn(48960, 1440).bfloat16()

    inputs = [input.clone() for _ in range(L)]

    mod = nn.GELU()

    niters = 500
    nwarmups = niters // 100

    for _ in range(nwarmups):
        for idx in range(L):
            out = mod(inputs[idx])

    t1 = time()
    for _ in range(niters):
        for idx in range(L):
            out = mod(inputs[idx])
    t2 = time()
    tt0 = (t2 - t1) / niters / L * 1000 #ms

    print("gelu, size", input.size(), "; time: {:.3f} ms".format(tt0))

#bench_gelu()


def bench_linear(M, N, K):

    input = torch.randn(M, K).bfloat16()
    weight = torch.randn(N, K).bfloat16()

    niters = 500
    nwarmups = niters // 100

    for _ in range(nwarmups):
        ref = torch.matmul(input, weight.t())

    t1 = time()
    for _ in range(niters):
        ref = torch.matmul(input, weight.t())
    t2 = time()

    for _ in range(nwarmups):
        out = weight_packed_linear(input, weight, None, True)

    t3 = time()
    for _ in range(niters):
        out = weight_packed_linear(input, weight, None, True)
    t4 = time()

    tt0 = (t2 - t1) / niters * 1000 # ms
    tt1 = (t4 - t3) / niters * 1000 # ms
    print(f"gemm_bf16(native): {tt0:.3f} ms, gemm_bf16(opt): {tt1:.3f} ms")

#bench_linear(48960, 1440, 1152)
#bench_linear(48960, 1152, 1440)

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, bias=True):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

def bench_linear_gelu_linear(batch_size, in_features, hidden_features, out_features):

    niters = 200
    nwarmups = niters // 10

    mod = MLP(in_features, hidden_features, out_features).to(dtype=torch.bfloat16)

    input = torch.randn(batch_size, in_features).to(dtype=torch.bfloat16)

    # weights
    w1 = convert_weight_packed(mod.fc1.weight)
    b1 = mod.fc1.bias.float()
    w2 = convert_weight_packed(mod.fc2.weight)
    b2 = mod.fc2.bias.float()

    #input.fill_(1)
    #w1.fill_(1)
    #b1.fill_(1)

    '''
    for _ in range(nwarmups):
        out = mod(input)
    '''

    t1 = time()
    '''
    for _ in range(niters):
        out = mod(input)
    '''
    t2 = time()

    tt0 = (t2 - t1) / niters * 1000 # ms

    for _ in range(nwarmups):
        out2 = fused_linear_gelu_linear(
            input,
            w1,
            w2,
            b1,
            b2,
            True,
            True)

    t3 = time()
    for _ in range(niters):
         out2 = fused_linear_gelu_linear(
            input,
            w1,
            w2,
            b1,
            b2,
            True,
            True)
    t4 = time()
    tt1 = (t4 - t3) / niters * 1000 # ms

    print(f"linear-gelu-linear(native): {tt0:.3f} ms; opt: {tt1:.3f} ms")

    #compare(out, out2)

bench_linear_gelu_linear(48960, 1152, 1440, 1152)
#bench_linear_gelu_linear(2, 32, 64, 32)


