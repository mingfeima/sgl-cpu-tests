import torch
from sgl_kernel.common_ops import qkv_proj_with_rope
from sgl_kernel.common_ops import convert_weight_packed

from utils import compare

torch.manual_seed(1111)


# constants
kv_lora_rank = 512
qk_head_dim = 192
qk_nope_head_dim = 128
qk_rope_head_dim = 64
rotary_dim = qk_rope_head_dim
num_heads = 22
q_lora_rank = 1536


def layernorm(x, weight, variance_epsilon=1e-6, residual=None):
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + variance_epsilon)
    return (x * weight).to(orig_dtype)

def _rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)

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

def rotary_emb(q_pe, k_pe, pos, cos_sin_cache):
    orig_dtype = q_pe.dtype
    q_pe = q_pe.float()
    k_pe = k_pe.float()
    cos_sin_cache = cos_sin_cache.float()

    query_rot = q_pe[..., : rotary_dim]
    key_rot = k_pe[..., : rotary_dim]
    cos_sin = cos_sin_cache[pos]
    cos, sin = cos_sin.chunk(2, dim=-1)
    cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
    sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)
    query_rot = query_rot * cos + _rotate_gptj(query_rot) * sin
    key_rot = key_rot * cos + _rotate_gptj(key_rot) * sin
    return query_rot.to(orig_dtype), key_rot.to(orig_dtype)

def native_torch(q_input, hidden_states, q_a_proj_weight, norm_weight1,
    q_b_proj_weight, w_kc, kv_a_proj_weight, norm_weight2, pos, cos_sin_cache):

    q = torch.matmul(hidden_states, q_a_proj_weight.t())
    q = layernorm(q, norm_weight1)
    q = torch.matmul(q, q_b_proj_weight.t()).view(-1, num_heads, qk_head_dim)

    q_nope, q_pe = q.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)
    q_nope_out = torch.bmm(q_nope.transpose(0, 1), w_kc)

    q_input[..., : kv_lora_rank] = q_nope_out.transpose(0, 1)
    latent_cache = torch.matmul(hidden_states, kv_a_proj_weight.t())
    v_input = latent_cache[..., : kv_lora_rank]
    v_input = layernorm(v_input.contiguous(), norm_weight2).unsqueeze(1)
    k_input = latent_cache.unsqueeze(1)
    k_input[..., : kv_lora_rank] = v_input
    k_pe = k_input[..., kv_lora_rank :]

    q_pe, k_pe = rotary_emb(q_pe, k_pe, pos, cos_sin_cache)
    q_input[..., kv_lora_rank :] = q_pe
    k_input[..., kv_lora_rank :] = k_pe

    return q_input, k_input, v_input

def native_torch_int8(q_input, hidden_states, w1_q, w1_s, norm_weight1,
        w2_q, w2_s, w_kc, w3_q, w3_s, norm_weight2, pos, cos_sin_cache):

    a_q, a_s = per_token_quant_int8(hidden_states)
    q = native_w8a8_per_token_matmul(a_q, w1_q, a_s, w1_s, torch.bfloat16)
    q = layernorm(q, norm_weight1)

    a_q, a_s = per_token_quant_int8(q)
    q = native_w8a8_per_token_matmul(a_q, w2_q, a_s, w2_s, torch.bfloat16).view(-1, num_heads, qk_head_dim)

    q_nope, q_pe = q.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)
    q_nope_out = torch.bmm(q_nope.transpose(0, 1), w_kc)

    q_input[..., : kv_lora_rank] = q_nope_out.transpose(0, 1)
    a_q, a_s = per_token_quant_int8(hidden_states)
    latent_cache = native_w8a8_per_token_matmul(a_q, w3_q, a_s, w3_s, torch.bfloat16)
    v_input = latent_cache[..., : kv_lora_rank]
    v_input = layernorm(v_input.contiguous(), norm_weight2).unsqueeze(1)
    k_input = latent_cache.unsqueeze(1)
    k_input[..., : kv_lora_rank] = v_input
    k_pe = k_input[..., kv_lora_rank :]

    q_pe, k_pe = rotary_emb(q_pe, k_pe, pos, cos_sin_cache)
    q_input[..., kv_lora_rank :] = q_pe
    k_input[..., kv_lora_rank :] = k_pe

    return q_input, k_input, v_input

def test_qkv_projection(B, hidden_size, dtype=torch.bfloat16):

    # [1, 7168]
    hidden_states = torch.randn(B, hidden_size, dtype=dtype) / hidden_size
    # [1, 22, 512 + 64]
    q_input = torch.empty(B, num_heads, kv_lora_rank + qk_rope_head_dim, dtype=dtype)
    # [1536, 7168]
    q_a_proj_weight = torch.randn(q_lora_rank, hidden_size, dtype=dtype) * 0.1
    # [1236] float32
    norm_weight1 = torch.randn(q_lora_rank, dtype=dtype)
    #[4224, 1536] or [22 * 192, 1536]
    q_b_proj_weight = torch.randn(num_heads * qk_head_dim, q_lora_rank, dtype=dtype) * 0.1
    # [22, 512, 128]
    w_kc = torch.randn(num_heads, kv_lora_rank, qk_nope_head_dim, dtype=dtype) * 0.1
    # [576, 7168]
    kv_a_proj_weight = torch.randn(kv_lora_rank + qk_rope_head_dim, hidden_size, dtype=dtype) * 0.1
    # []
    norm_weight2 = torch.randn(kv_lora_rank, dtype=dtype)
    # [1]
    pos = torch.randint(10, 100, (B,))
    # [max, 64]
    cos_sin_cache = torch.randn(100, rotary_dim, dtype=dtype)
    
    ## w_kc needs to be transposed to [H, IC, OC]
    q_input, k_input, v_input = native_torch(q_input, hidden_states, q_a_proj_weight, norm_weight1,
        q_b_proj_weight, w_kc.transpose(1, 2), kv_a_proj_weight, norm_weight2, pos, cos_sin_cache)

    ## w_kc passed in [H, OC, IC]
    eps = 1e-6
    qa_packed = convert_weight_packed(q_a_proj_weight)
    qb_packed = convert_weight_packed(q_b_proj_weight)
    kva_packed = convert_weight_packed(kv_a_proj_weight)
    wkc_packed = convert_weight_packed(w_kc)

    # bfloat16
    q_input2, k_input2, v_input2 = qkv_proj_with_rope(hidden_states, qa_packed,
        qb_packed, kva_packed, wkc_packed, norm_weight1, norm_weight2, pos, cos_sin_cache, eps,
        False, None, None, None, True)
  
    print("Compare Ref and C++ on bfloat16:")
    compare(q_input.narrow(2, 512, 64), q_input2.narrow(2, 512, 64))
    compare(q_input, q_input2)
    compare(k_input, k_input2)
    compare(v_input, v_input2)

    # int8 w8a8
    w1_q, w1_s = per_token_quant_int8(q_a_proj_weight)
    w2_q, w2_s = per_token_quant_int8(q_b_proj_weight)
    w3_q, w3_s = per_token_quant_int8(kv_a_proj_weight)
    q_input3 = torch.empty(B, num_heads, kv_lora_rank + qk_rope_head_dim, dtype=dtype)
    q_input3, k_input3, v_input3 = native_torch_int8(q_input3, hidden_states, w1_q, w1_s, norm_weight1,
        w2_q, w2_s, w_kc.transpose(1, 2), w3_q, w3_s, norm_weight2, pos, cos_sin_cache)

    print("\nCompare Ref bfloat16 and Ref int8:")
    compare(q_input.narrow(2, 512, 64), q_input3.narrow(2, 512, 64))
    compare(q_input, q_input3)
    compare(k_input, k_input3)
    compare(v_input, v_input3)

    w1_q_packed = convert_weight_packed(w1_q)
    w2_q_packed = convert_weight_packed(w2_q)
    w3_q_packed = convert_weight_packed(w3_q)

    print("\nCompare Ref int8 and C++ int8:")
    q_input4, k_input4, v_input4 = qkv_proj_with_rope(hidden_states, w1_q_packed,
        w2_q_packed, w3_q_packed, wkc_packed, norm_weight1, norm_weight2, pos, cos_sin_cache, eps,
        True, w1_s, w2_s, w3_s, True)

    compare(q_input3.narrow(2, 512, 64), q_input4.narrow(2, 512, 64))
    compare(q_input3, q_input4)
    compare(k_input3, k_input4)
    compare(v_input3, v_input4)

test_qkv_projection(3, 7168)
