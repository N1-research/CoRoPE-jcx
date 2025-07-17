import triton
import triton.language as tl
import torch

@triton.jit
def co_rope_bwd_kernel(
    Q, K, V, O, M, L, a_k_out,  # forward inputs/outputs
    dO,                # gradient w.r.t. output
    dQ, dK, dV, dThetas,        # gradients to compute
    thetas,
    stride_batch, stride_head, stride_seq, stride_dim,
    seq_len: tl.constexpr,
    SM_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
    CAUSAL_MASK_VALUE: tl.constexpr,
):
    m_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)
    bid = batch_head_idx // NUM_HEADS
    hid = batch_head_idx % NUM_HEADS

    # Compute offsets
    qkv_offset = batch_head_idx.to(tl.int64) * stride_head
    Q_base = Q + qkv_offset
    K_base = K + qkv_offset
    V_base = V + qkv_offset
    dQ_base = dQ + qkv_offset
    dK_base = dK + qkv_offset
    dV_base = dV + qkv_offset
    O_base = O + qkv_offset
    dO_base = dO + qkv_offset
    a_k_ptr = a_k_out + batch_head_idx * seq_len

    offs_m = m_idx
    mask_m = offs_m < seq_len

    # Load q, k, v, o, do for this position
    q_ptrs = Q_base + offs_m * stride_seq + tl.arange(0, HEAD_DIM)
    k_ptrs = K_base + offs_m * stride_seq + tl.arange(0, HEAD_DIM)
    v_ptrs = V_base + offs_m * stride_seq + tl.arange(0, HEAD_DIM)
    o_ptrs = O_base + offs_m * stride_seq + tl.arange(0, HEAD_DIM)
    do_ptrs = dO_base + offs_m * stride_seq + tl.arange(0, HEAD_DIM)

    q = tl.load(q_ptrs, mask=mask_m, other=0.0).to(tl.float32)
    k = tl.load(k_ptrs, mask=mask_m, other=0.0).to(tl.float32)
    v = tl.load(v_ptrs, mask=mask_m, other=0.0).to(tl.float32)
    o = tl.load(o_ptrs, mask=mask_m, other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask_m, other=0.0).to(tl.float32)

    # Load thetas for this head
    theta_ptr = thetas + hid * HEAD_DIM + tl.arange(0, HEAD_DIM)
    theta_this_head = tl.load(theta_ptr).to(tl.float32)

    # Load a_k values for this query
    a_k_vals = tl.load(a_k_ptr + tl.arange(0, seq_len))  # [seq_len]

    # Initialize gradients
    dq = tl.zeros([HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([HEAD_DIM], dtype=tl.float32)
    dtheta = tl.zeros([HEAD_DIM], dtype=tl.float32)

    # Compute rotary embeddings for q
    a_k_last = tl.load(a_k_ptr + (seq_len - 1))
    freqs_q = a_k_last * theta_this_head
    freqs_cos_q = tl.cos(freqs_q)
    freqs_sin_q = tl.sin(freqs_q)

    idx = tl.arange(0, HEAD_DIM)
    even_mask = (idx % 2 == 0)[None, :]
    odd_mask = (idx % 2 == 1)[None, :]
    swap_offsets = idx + 1
    swap_mask = swap_offsets < HEAD_DIM

    q_rot = (
        even_mask * (freqs_cos_q * q - freqs_sin_q * tl.load(q_ptrs + swap_offsets, mask=swap_mask, other=0.0))
        + odd_mask * (freqs_sin_q * tl.load(q_ptrs + swap_offsets, mask=swap_mask, other=0.0) + freqs_cos_q * q)
    )

    # Process blocks for attention computation
    for start_n in tl.range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len

        k_ptrs_block = K_base + offs_n[:, None] * stride_seq + tl.arange(0, HEAD_DIM)[None, :]
        v_ptrs_block = V_base + offs_n[:, None] * stride_seq + tl.arange(0, HEAD_DIM)[None, :]
        k_block = tl.load(k_ptrs_block, mask=mask_n[:, None], other=0.0).to(tl.float32)
        v_block = tl.load(v_ptrs_block, mask=mask_n[:, None], other=0.0)

        # Compute rotary embeddings for k
        a_k_block = tl.load(a_k_ptr + offs_n)  # [BLOCK_N]
        freqs_k = a_k_block[:, None] * theta_this_head[None, :]
        freqs_cos_k = tl.cos(freqs_k)
        freqs_sin_k = tl.sin(freqs_k)

        k_rot = (
            even_mask * (freqs_cos_k * k_block - freqs_sin_k * tl.load(k_ptrs_block + swap_offsets[None, :], mask=mask_n[:, None] & swap_mask[None, :], other=0.0))
            + odd_mask * (freqs_sin_k * tl.load(k_ptrs_block + swap_offsets[None, :], mask=mask_n[:, None] & swap_mask[None, :], other=0.0) + freqs_cos_k * k_block)
        )

        # Compute attention logits
        logits = tl.sum(q_rot * k_rot, axis=1) * SM_SCALE

        # Apply causal mask if needed
        if CAUSAL:
            causal_mask = offs_m >= offs_n
            attn_mask = causal_mask & mask_n
        else:
            attn_mask = mask_n

        logits = tl.where(attn_mask, logits, CAUSAL_MASK_VALUE)

        # Compute attention weights (softmax)
        m_max = tl.max(logits, axis=0)
        logits_stable = logits - m_max
        exp_logits = tl.exp(logits_stable)
        sum_exp = tl.sum(exp_logits, axis=0)
        attention_weights = exp_logits / sum_exp

        # Gradient computation for attention weights
        # ∂L/∂attention_weights = do * v
        d_attention = do * v_block  # [BLOCK_N, HEAD_DIM]

        # Simplified gradient for logits (diagonal only)
        d_logits = tl.sum(d_attention, axis=1) * attention_weights * (1 - attention_weights)

        # Gradient for q_rot and k_rot
        # ∂L/∂q_rot = ∂L/∂logits * k_rot
        # ∂L/∂k_rot = ∂L/∂logits * q_rot
        dq_rot = tl.sum(d_logits[:, None] * k_rot, axis=0)
        dk_rot = d_logits[:, None] * q_rot[None, :]

        # Gradient for q (unrotate)
        # ∂L/∂q = ∂L/∂q_rot * ∂q_rot/∂q
        dq += dq_rot * (even_mask * freqs_cos_q - odd_mask * freqs_sin_q + 
                        even_mask * freqs_sin_q + odd_mask * freqs_cos_q)

        # Gradient for k (unrotate)
        # ∂L/∂k = ∂L/∂k_rot * ∂k_rot/∂k
        dk_unrot = dk_rot * (even_mask * freqs_cos_k - odd_mask * freqs_sin_k + 
                             even_mask * freqs_sin_k + odd_mask * freqs_cos_k)
        
        # Accumulate dk for all positions
        dk += tl.sum(dk_unrot * mask_n[:, None], axis=0)

        # Gradient for v
        # ∂L/∂v = ∂L/∂attention * attention_weights
        dv += tl.sum(d_attention * attention_weights[:, None], axis=0)

        # Gradient for thetas
        # ∂L/∂theta = ∂L/∂q_rot * ∂q_rot/∂theta + ∂L/∂k_rot * ∂k_rot/∂theta
        dtheta_q = dq_rot * q * a_k_last * (-even_mask * freqs_sin_q - odd_mask * freqs_cos_q + 
                                            even_mask * freqs_cos_q - odd_mask * freqs_sin_q)
        dtheta += dtheta_q

        # For k rotations
        # Compute dtheta_k for all i in the block
        dtheta_k = dk_rot * k_block * a_k_block[:, None] * (
            -even_mask * freqs_sin_k - odd_mask * freqs_cos_k +
            even_mask * freqs_cos_k - odd_mask * freqs_sin_k
        )  # shape: [BLOCK_N, HEAD_DIM]

        # Mask out invalid rows and sum
        dtheta += tl.sum(dtheta_k * mask_n[:, None], axis=0)

    # Store gradients
    tl.store(dQ_base + offs_m * stride_seq + tl.arange(0, HEAD_DIM), dq, mask=mask_m)
    tl.store(dK_base + offs_m * stride_seq + tl.arange(0, HEAD_DIM), dk, mask=mask_m)
    tl.store(dV_base + offs_m * stride_seq + tl.arange(0, HEAD_DIM), dv, mask=mask_m)

    if m_idx == 0:  
        dtheta_ptr = dThetas + hid * HEAD_DIM + tl.arange(0, HEAD_DIM)
        tl.store(dtheta_ptr, dtheta)

def co_rope_backward(q, k, v, o, m, l, a_k_out, do, thetas, causal, sm_scale):
    batch_size, num_heads, seq_len, head_dim = q.shape

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dthetas = torch.zeros_like(thetas)

    def grid(meta):
        return (seq_len, batch_size * num_heads)

    co_rope_bwd_kernel[grid](
        Q=q.contiguous(),
        K=k.contiguous(),
        V=v.contiguous(),
        O=o.contiguous(),
        M=m.contiguous(),
        L=l.contiguous(),
        a_k_out=a_k_out.contiguous(),
        dO=do.contiguous(),
        dQ=dq,
        dK=dk,
        dV=dv,
        dThetas=dthetas,
        thetas=thetas,
        stride_batch=q.stride(0),
        stride_head=q.stride(1),
        stride_seq=q.stride(2),
        stride_dim=q.stride(3),
        seq_len=seq_len,
        SM_SCALE=sm_scale,
        BLOCK_M=1,
        BLOCK_N=64,
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        CAUSAL=causal,
        CAUSAL_MASK_VALUE=-torch.finfo(torch.float32).max,
    )
    return dq, dk, dv, dthetas