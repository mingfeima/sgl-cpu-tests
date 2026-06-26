import os
from time import time

import torch
import torch.nn.functional as F

import sgl_kernel  # noqa: F401  # load torch.ops.sgl_kernel


torch.manual_seed(1111)

DTYPES = [torch.bfloat16, torch.float16]
SHAPES = [
    # Qwen3.5 CPU decode MLP activation shape observed in torch profiler.
    (1, 18432),
    # Qwen3.5 one-batch prefill input_len=1000 equivalent for the same activation.
    (1000, 18432),
    # Neighboring non-power-of-two token count to avoid only testing ideal loops.
    (17, 18432),
]


def ref_silu_and_mul(x):
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def bench_one(shape, dtype):
    x = torch.randn(shape, dtype=dtype)
    ref = ref_silu_and_mul(x)
    out = torch.ops.sgl_kernel.silu_and_mul_cpu(x)
    torch.testing.assert_close(out, ref, atol=1e-2 if dtype is torch.bfloat16 else 1e-3, rtol=1e-2 if dtype is torch.bfloat16 else 1e-3)

    # Keep total measured work roughly stable across decode/prefill shapes.
    num_tokens = shape[0]
    niters = int(os.environ.get("NITERS", 20000 if num_tokens == 1 else 2000 if num_tokens < 128 else 100))
    warmup = int(os.environ.get("WARMUP", min(20, niters // 10)))

    for _ in range(warmup):
        torch.ops.sgl_kernel.silu_and_mul_cpu(x)

    t0 = time()
    for _ in range(niters):
        torch.ops.sgl_kernel.silu_and_mul_cpu(x)
    t1 = time()

    us = (t1 - t0) * 1_000_000 / niters
    elems = x.numel()
    ns_per_elem = us * 1000 / elems
    print(
        f"### silu_and_mul_cpu: shape={shape}, dtype={dtype}, "
        f"niters={niters}, latency={us:.3f} us, ns/elem={ns_per_elem:.4f}"
    )


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", torch.get_num_threads())))
    print(f"torch_num_threads={torch.get_num_threads()}")
    for dtype in DTYPES:
        for shape in SHAPES:
            bench_one(shape, dtype)


if __name__ == "__main__":
    main()
