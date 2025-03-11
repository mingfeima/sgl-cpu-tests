import torch
from torch.nn.functional import scaled_dot_product_attention
from sgl_kernel.ops._kernels import extend_attention_cpu as extend_attention
from utils import compare

torch.manual_seed(1111)

def _run_sdpa_forward_extend(
    query: torch.Tensor,
    output: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_prefix_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    scaling=None,
    enable_gqa=False,
    causal=False,
):

    assert seq_lens.shape[0] == extend_prefix_lens.shape[0]
    assert seq_lens.shape[0] == extend_seq_lens.shape[0]

    # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
    query = query.movedim(0, query.dim() - 2)

    start_q, start_kv = 0, 0
    for seq_idx in range(seq_lens.shape[0]):
        # TODO: this loop process a sequence per iter, this is inefficient.
        # Need optimize the performance later.

        extend_seq_len_q = extend_seq_lens[seq_idx]
        prefill_seq_len_q = extend_prefix_lens[seq_idx]

        seq_len_kv = seq_lens[seq_idx]
        end_q = start_q + extend_seq_len_q
        end_kv = start_kv + seq_len_kv

        per_req_query = query[:, start_q:end_q, :]
        per_req_query_redudant = torch.empty(
            (per_req_query.shape[0], seq_len_kv, per_req_query.shape[2]),
            dtype=per_req_query.dtype,
            device=per_req_query.device,
        )

        per_req_query_redudant[:, prefill_seq_len_q:, :] = per_req_query

        # get key and value from cache. per_req_tokens contains the kv cache
        # index for each token in the sequence.
        req_pool_idx = req_pool_indices[seq_idx]
        per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
        per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
        per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

        print("### per_req_key: ", per_req_key.size())
        print("per_req_value: ", per_req_value.size())

        per_req_out_redudant = (
            scaled_dot_product_attention(
                per_req_query_redudant.unsqueeze(0),
                per_req_key.unsqueeze(0),
                per_req_value.unsqueeze(0),
                enable_gqa=enable_gqa,
                scale=scaling,
                is_causal=causal,
            )
            .squeeze(0)
            .movedim(query.dim() - 2, 0)
        )
        output[start_q:end_q, :, :] = per_req_out_redudant[prefill_seq_len_q:, :, :]
        start_q, start_kv = end_q, end_kv
    return output


def test_extend_attention_once(B, N_CTX, H_Q, H_KV, D, DV, mla=False):
    dtype = torch.bfloat16

    b_seq_len_prefix = torch.randint(1, N_CTX // 2, (B,), dtype=torch.int32)
    if mla:
        b_seq_len_prefix.zero_()
    b_seq_len_extend = torch.randint(1, N_CTX // 2, (B,), dtype=torch.int32)
    b_seq_len = (b_seq_len_prefix + b_seq_len_extend)
    max_len_in_batch = torch.max(b_seq_len, 0)[0].item()
    print("b_seq_len_prefix: ", b_seq_len_prefix)
    print("b_seq_len_extend: ", b_seq_len_extend)
    print("b_seq_len       : ", b_seq_len)
    print("max_len_in_batch: ", max_len_in_batch)

    b_req_idx = torch.arange(B, dtype=torch.int32)
    req_to_tokens = torch.empty((B, max_len_in_batch), dtype=torch.int32)
    b_start_loc = torch.zeros((B,), dtype=torch.int32)
    b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], 0)
    b_start_loc_extend = torch.zeros((B,), dtype=torch.int32)
    b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)
    print("b_start_loc     : ", b_start_loc)
    print("b_start_loc_extend: ", b_start_loc_extend)

    for i in range(B):
        req_to_tokens[i, : b_seq_len[i]] = torch.arange(
            b_start_loc[i], b_start_loc[i] + b_seq_len[i]
        )

    total_token_num = torch.sum(b_seq_len).item()
    extend_token_num = torch.sum(b_seq_len_extend).item()
    k_buffer = torch.randn((total_token_num, H_KV, D), dtype=dtype)
    v_buffer = torch.randn((total_token_num, H_KV, DV), dtype=dtype)

    k_extend = torch.empty((extend_token_num, H_KV, D), dtype=dtype)
    v_extend = torch.empty((extend_token_num, H_KV, DV), dtype=dtype)
    q_extend = torch.empty((extend_token_num, H_Q, D), dtype=dtype)

    # trick for debugging flash attn
    # for i in range(total_token_num):
    #     k_buffer[i].fill_(i)
    #     v_buffer[i].fill_(i)


    for i in range(B):
        extend_start_in_buffer = b_start_loc[i] + b_seq_len_prefix[i]
        extend_end_in_buffer = b_start_loc[i] + b_seq_len[i]
        extend_start = b_start_loc_extend[i]
        extend_end = b_start_loc_extend[i] + b_seq_len_extend[i]
        k_extend[extend_start:extend_end] = k_buffer[extend_start_in_buffer:extend_end_in_buffer]
        v_extend[extend_start:extend_end] = v_buffer[extend_start_in_buffer:extend_end_in_buffer]
        q_extend[extend_start:extend_end] = torch.randn((b_seq_len_extend[i], H_Q, D), dtype=dtype)

    #q_extend.fill_(1)

    b_seq_len_extend = b_seq_len - b_seq_len_prefix
    b_start_loc_extend = torch.zeros_like(b_seq_len)
    b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)
    max_len_extend = torch.max(b_seq_len_extend, 0)[0].item()

    print("b_start_loc_extend: ", b_start_loc_extend)
    print("max_len_extend :", max_len_extend)

    sm_scale = 1.0 / (D**0.5)
    logit_cap = 0.0

    # handle index type
    b_req_idx = b_req_idx.to(torch.int64)
    b_seq_len = b_seq_len.to(torch.int64)

    enable_gqa = H_Q != H_KV
    o_ref = torch.empty((extend_token_num, H_Q, DV), dtype=dtype)
    _run_sdpa_forward_extend(
       q_extend,
       o_ref,
       k_buffer,
       v_buffer,
       req_to_tokens,
       b_req_idx,
       b_seq_len,
       b_seq_len_prefix,
       b_seq_len_extend,
       scaling = sm_scale,
       enable_gqa = enable_gqa,
       causal = True
    )

    o_extend = torch.empty((extend_token_num, H_Q, DV), dtype=dtype)
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

    #print(o_ref, o_ref.size())
    #print(o_extend, o_extend.size())

    #print(o_ref[:, 0, 0].view(-1))
    #print(o_extend[:, 0, 0].view(-1))

    compare(o_ref, o_extend)

for is_mla in [True, False]:
    test_extend_attention_once(1, 123, 1, 1, 128, 96, is_mla)
    #test_extend_attention_once(1, 123, 16, 1, 128, 96, is_mla)
    test_extend_attention_once(4, 1230, 16, 4, 128, 96, is_mla)
