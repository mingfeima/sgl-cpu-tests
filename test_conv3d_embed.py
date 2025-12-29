import torch
from torch.nn.functional import scaled_dot_product_attention

import sgl_kernel
conv3d_embed = torch.ops.sgl_kernel.conv3d_embed_cpu
conv3d_embed_weight_pack = torch.ops.sgl_kernel.conv3d_embed_weight_pack
weight_packed_linear = torch.ops.sgl_kernel.weight_packed_linear

from utils import compare

from time import time

torch.manual_seed(1234)


def conv3d_embed_ref(input, weight, bias):

    M = input.size(0)
    K = input.numel() // M // 2
    N = weight.size(0)

    input0 = input[:,:,0,:,:].contiguous().view(M, K).float()
    weight0 = weight[:,:,0,:,:].contiguous().view(N, K).float()
    weight1 = weight[:,:,1,:,:].contiguous().view(N, K).float()

    out0 = torch.matmul(input0, weight0.t())
    out1 = torch.matmul(input0, weight1.t())

    return (out0 + out1).add_(bias.float()).bfloat16()


def test_conv3d_prepack():

    OC = 1152
    IC = 3
    D = 2
    H = 16
    W = 16

    BLOCK_N = 32

    weight = torch.randn(OC, IC, D, H, W).bfloat16()

    packed = conv3d_embed_weight_pack(weight)

    ref = weight.view(OC // BLOCK_N, BLOCK_N, IC, D, H * W).permute(0, 2, 3, 1, 4).contiguous()
    ref = ref.view(-1, BLOCK_N, H * W // 2, 2).transpose(1, 2).contiguous()

    compare(ref.view(-1), packed.view(-1), True)

#test_conv3d_prepack()


def test_conv3d_embed(N):

    Cin = 3
    Cout = 1152

    # repeat on T dim
    input = torch.randn(N, Cin, 16, 16).unsqueeze(2).repeat(1, 1, 2, 1, 1).bfloat16()

    conv = torch.nn.Conv3d(
        Cin,
        Cout,
        kernel_size=[2, 16, 16],
        stride=[2, 16, 16],
        bias=True).bfloat16()

    weight = conv.weight
    bias = conv.bias

    out_ref = conv(input)

    out_torch = conv3d_embed_ref(input, weight, bias).view(out_ref.shape)

    out = conv3d_embed(input, weight, bias, False)

    compare(out_ref, out_torch)
    compare(out_ref, out.view(out_ref.shape))

    niters = 400
    nwarmups = niters // 10

    #x = input.view(N, -1)
    #w = weight.view(Cout, -1)
    #bias = bias.float()

    for _ in range(nwarmups):
        with torch.no_grad():
            #out_ref = conv(input)
            #o = torch.matmul(x, w.t()).add_(bias)
            #o = weight_packed_linear(x, w, bias, True)
            conv3d_embed(input, weight, bias, True)

    t1 = time()
    for _ in range(niters):
        with torch.no_grad():
            #out_ref = conv(input)
            #o = torch.matmul(x, w.t()).add_(bias)
            #o = weight_packed_linear(x, w, bias, True)
            conv3d_embed(input, weight, bias, True)
    t2 = time()

    tt0 = (t2 - t1) / niters * 1000 #ms

    print(f"### Conv3d N {N}, Cin {Cin}, Cout {Cout}: {tt0:.3f} ms")

test_conv3d_embed(48960)
