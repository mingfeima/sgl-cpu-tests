import torch

pres = {
    torch.bfloat16 : 1e-2,
    torch.float16: 1e-3,
    torch.float32: 1e-5,
}

def compare(a: torch.Tensor, b: torch.Tensor, debug=False):

    atol = rtol = pres[a.dtype]

    res = torch.allclose(a, b, rtol=rtol, atol=atol)

    max_diff = (a - b).abs().max().item()
    max_index = torch.argmax((a-b).abs()).item()
    a_sum = a.sum().item()
    b_sum = b.sum().item()

    if debug:
        print(a)
        print(b)
        print(max_index, a.flatten()[max_index], b.flatten()[max_index])

    print("Comparing: ", res, " max_diff = {:.5f}, asum = {:.3f}, bsum = {:.3f}".format(max_diff, a_sum, b_sum))
