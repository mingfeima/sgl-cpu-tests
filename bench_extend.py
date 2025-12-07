import torch
import sgl_kernel
extend_attention = torch.ops.sgl_kernel.extend_attention_cpu
from time import time

torch.manual_seed(1111)


# N_CTX is the exact context length for MLA
# otherwise prefix and extend take fifty-fifty
def bench_extend_attention_once(B, N_CTX, H_Q, H_KV, D, DV, mla=False):

    niters = 1000

    # scale `niters` for larger context length
    scale = max(1, N_CTX // 200)
    niters = niters // scale
    nwarmups = max(5, niters // 100)

    dtype = torch.bfloat16

    b_seq_len_prefix = torch.full((B,), 0 if mla else N_CTX // 2, dtype=torch.int32)
    b_seq_len_extend = torch.full((B,), N_CTX if mla else N_CTX // 2, dtype=torch.int32)
    b_seq_len = (b_seq_len_prefix + b_seq_len_extend)
    max_len_in_batch = torch.max(b_seq_len, 0)[0].item()

    b_req_idx = torch.arange(B, dtype=torch.int32)
    req_to_tokens = torch.empty((B, max_len_in_batch), dtype=torch.int32)
    b_start_loc = torch.zeros((B,), dtype=torch.int32)
    b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], 0)
    b_start_loc_extend = torch.zeros((B,), dtype=torch.int32)
    b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)

    for i in range(B):
        req_to_tokens[i, : b_seq_len[i]] = torch.arange(
            b_start_loc[i], b_start_loc[i] + b_seq_len[i]
        )

    total_token_num = torch.sum(b_seq_len).item()
    extend_token_num = torch.sum(b_seq_len_extend).item()

    H_BUF = 1 if mla else H_KV
    k_buffer = torch.randn((total_token_num, H_BUF, D), dtype=dtype)
    v_buffer = torch.randn((total_token_num, H_BUF, DV), dtype=dtype)

    k_extend = torch.empty((extend_token_num, H_KV, D), dtype=dtype)
    v_extend = torch.empty((extend_token_num, H_KV, DV), dtype=dtype)
    q_extend = torch.empty((extend_token_num, H_Q, D), dtype=dtype)

    for i in range(B):
        extend_start_in_buffer = b_start_loc[i] + b_seq_len_prefix[i]
        extend_end_in_buffer = b_start_loc[i] + b_seq_len[i]
        extend_start = b_start_loc_extend[i]
        extend_end = b_start_loc_extend[i] + b_seq_len_extend[i]
        k_extend[extend_start:extend_end] = k_buffer[extend_start_in_buffer:extend_end_in_buffer]
        v_extend[extend_start:extend_end] = v_buffer[extend_start_in_buffer:extend_end_in_buffer]
        q_extend[extend_start:extend_end] = torch.randn((b_seq_len_extend[i], H_Q, D), dtype=dtype)

    max_len_extend = torch.max(b_seq_len_extend, 0)[0].item()

    sm_scale = 1.0 / (D**0.5)
    logit_cap = 0.0

    # handle index type
    b_req_idx = b_req_idx.to(torch.int64)
    b_seq_len = b_seq_len.to(torch.int64)

    o_extend = torch.empty((extend_token_num, H_Q, DV), dtype=dtype)
    for _ in range(nwarmups):
        extend_attention(
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer,
            v_buffer,
            req_to_tokens,
            b_req_idx,
            b_seq_len,
            b_seq_len_extend,
            b_start_loc_extend,
            max_len_extend,
            sm_scale,
            logit_cap)

    t0 = time()
    for _ in range(niters):
        extend_attention(
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer,
            v_buffer,
            req_to_tokens,
            b_req_idx,
            b_seq_len,
            b_seq_len_extend,
            b_start_loc_extend,
            max_len_extend,
            sm_scale,
            logit_cap)
    t1 = time()
    tt0 = (t1 - t0) / niters * 1000 # ms
    print(f"### extend_attention: seqlen {N_CTX}: {tt0:.3f} ms")


# bench_extend_attention_once(B, N_CTX, H_Q, H_KV, D, DV, mla=False)
for N_CTX in [128, 256, 500, 1000, 2000, 3500, 6400, 8000]:
    bench_extend_attention_once(4, N_CTX, 22, 22, 192, 128, True)
