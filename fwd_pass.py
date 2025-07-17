import torch
import math
import sys
import os

from CoROPE.co_rope_forward import co_rope_forward

def print_red_warning(message):
    print(f"\033[31mWARNING: {message}\033[0m")

def calc_sim(x, y, name="tensor"):
    x, y = x.data.double(), y.data.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        print_red_warning(f'{name} all zero')
        return 1.0
    sim = 2 * (x * y).sum() / denominator
    return sim.item()

def assert_similar(x, y, eps=1e-4, name="tensor"):
    x, y = x.data.double(), y.data.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        print(f"WARNING: {name} all zero")
        return
    sim = 2 * (x * y).sum() / denominator
    diff = 1. - sim.item()
    if not (0 <= diff <= eps):
        print(f"\033[31m{name} Error: {diff}\033[0m")
    else:
        print(f'passed: {name} diff={diff:.2e}')

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(x, sin, cos):
    return (x * cos) + (rotate_half(x) * sin)

def get_fixed_sin_cos(seq_len, dim, device):
    position = torch.arange(seq_len, device=device).float()
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    sinusoid_inp = torch.einsum('i,j->ij', position, inv_freq)
    sin = torch.sin(sinusoid_inp)
    cos = torch.cos(sinusoid_inp)
    sin = torch.repeat_interleave(sin, 2, dim=-1)
    cos = torch.repeat_interleave(cos, 2, dim=-1)
    return sin, cos

def co_rope_forward_pairwise_torch(q, k, v):
    # q, k, v: [B, H, T, D]
    B, H, T, D = q.shape
    device = q.device
    sin, cos = get_fixed_sin_cos(T, D, device)
    sin = sin.unsqueeze(0).unsqueeze(0)
    cos = cos.unsqueeze(0).unsqueeze(0)

    logits = torch.einsum('bhid,bhjd->bhij', q, k)
    sig_logits = torch.sigmoid(logits)

    mask = torch.tril(torch.ones(T, T, device=device))
    a = torch.cumsum(sig_logits * mask, dim=-1).unsqueeze(-1)  # [B, H, T, T, 1]

    sin_exp = sin.expand(B, H, T, D)
    cos_exp = cos.expand(B, H, T, D)

    # q 用 a_t
    a_q = torch.diagonal(a.squeeze(-1), dim1=-2, dim2=-1).unsqueeze(-1)  # [B, H, T, 1]
    sin_q = sin_exp * a_q
    cos_q = cos_exp * a_q
    q_rot = apply_rotary_pos_emb(q, sin_q, cos_q)
    sin_k = sin_exp.unsqueeze(3) * a  # [B, H, T, T, D]
    cos_k = cos_exp.unsqueeze(3) * a
    k_exp = k.unsqueeze(2).expand(-1, -1, T, -1, -1)
    k_rot = apply_rotary_pos_emb(k_exp, sin_k, cos_k)
    attn_weights = (q_rot.unsqueeze(3) * k_rot).sum(dim=-1) / math.sqrt(D)
    attn_probs = torch.softmax(attn_weights, dim=-1)
    attn_output = torch.einsum('bhts,bhsd->bhtd', attn_probs, v)
    return attn_output  

def co_rope_forward_torch_exact(q, k, v, causal, sm_scale, theta=10000.0):
    batch_size, num_heads, seq_len, head_dim = q.shape
    device = q.device

    # Precompute thetas
    thetas = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    thetas = thetas.repeat_interleave(2)
    thetas = thetas.unsqueeze(0).repeat(num_heads, 1).contiguous()  # [H, D]

    o = torch.zeros_like(q)
    m = torch.full((batch_size, num_heads, seq_len), -1e9, device=device, dtype=torch.float32)
    l = torch.zeros((batch_size, num_heads, seq_len), device=device, dtype=torch.float32)
    a_k_out = torch.zeros((batch_size, num_heads, seq_len), device=device, dtype=torch.float32)

    for b in range(batch_size):
        for h in range(num_heads):
            for t in range(seq_len):
                # Step 1: Compute prefix sum a_k for this query
                a_k = torch.zeros(seq_len, device=device, dtype=torch.float32)
                prev = 0.0
                for idx in range(seq_len):
                    k_i = k[b, h, idx, :].float()
                    q_i = q[b, h, t, :].float()
                    dot = torch.dot(q_i, k_i)
                    prev = prev + dot
                    a_k[idx] = prev
                a_k_out[b, h, :] = a_k

                # Step 2: Rotary for q
                a_k_last = a_k[-1]
                freqs_q = a_k_last * thetas[h]
                freqs_cos_q = torch.cos(freqs_q)
                freqs_sin_q = torch.sin(freqs_q)
                q_vec = q[b, h, t, :].float()
                even_mask = (torch.arange(head_dim, device=device) % 2 == 0)
                odd_mask = ~even_mask
                q_rot = (
                    even_mask * q_vec * freqs_cos_q - odd_mask * q_vec * freqs_sin_q
                    + even_mask * q_vec * freqs_sin_q + odd_mask * q_vec * freqs_cos_q
                )

                o_acc = torch.zeros(head_dim, device=device, dtype=torch.float32)
                l_acc = 0.0
                m_acc = -1e9

                for start_n in range(0, seq_len, 64):
                    offs_n = torch.arange(start_n, min(start_n + 64, seq_len), device=device)
                    mask_n = offs_n < seq_len

                    k_block = k[b, h, offs_n, :].float()
                    v_block = v[b, h, offs_n, :].float()
                    a_k_vals = a_k[offs_n]
                    freqs_k = a_k_vals[:, None] * thetas[h][None, :]
                    freqs_cos_k = torch.cos(freqs_k)
                    freqs_sin_k = torch.sin(freqs_k)
                    even_mask = (torch.arange(head_dim, device=device) % 2 == 0)
                    odd_mask = ~even_mask
                    k_rot = (
                        even_mask * k_block * freqs_cos_k - odd_mask * k_block * freqs_sin_k
                        + even_mask * k_block * freqs_sin_k + odd_mask * k_block * freqs_cos_k
                    )
                    logits = (q_rot * k_rot).sum(dim=1) * sm_scale

                    if causal:
                        causal_mask = (t >= offs_n)
                        attn_mask = causal_mask & mask_n
                    else:
                        attn_mask = mask_n

                    logits = torch.where(attn_mask, logits, torch.tensor(-torch.finfo(torch.float32).max, device=device))
                    m_new = max(m_acc, logits.max().item())
                    alpha = math.exp(m_acc - m_new)
                    p = torch.exp(logits - m_new)
                    o_acc = o_acc * alpha + (p[:, None] * v_block).sum(dim=0)
                    l_acc = l_acc * alpha + p.sum()
                    m_acc = m_new

                o_acc = o_acc / l_acc
                o[b, h, t, :] = o_acc
                m[b, h, t] = m_acc
                l[b, h, t] = l_acc

    return o, a_k_out

if __name__ == "__main__":
    B, H, S, D = 1, 16, 512, 128 
    dtype = torch.float32

    torch.manual_seed(0)

    q = torch.randn(B, H, S, D, device='cuda', dtype=dtype)
    k = torch.randn(B, H, S, D, device='cuda', dtype=dtype)
    v = torch.randn(B, H, S, D, device='cuda', dtype=dtype)

    scale = 1.0 / math.sqrt(D)

    print("Running PyTorch reference (Triton-aligned)...")
    o_ref, a_k_ref = co_rope_forward_torch_exact(q, k, v, causal=True, sm_scale=scale)

    print("Running Triton kernel...")
    (_, _, _, _, _, _, _), o_tri = co_rope_forward(q, k, v, causal=True, sm_scale=scale)

    print("\n--- Numerical comparison ---")
    assert_similar(o_tri, o_ref, eps=1e-4, name="output")
