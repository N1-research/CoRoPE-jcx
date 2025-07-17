import torch
import triton
import triton.language as tl
@triton.jit
def co_rope_fwd_kernel(
    Q, K, V, O, M, L,
    thetas,  # [NUM_HEADS, HEAD_DIM]
    stride_batch, stride_head, stride_seq, stride_dim,
    seq_len: tl.constexpr,
    SM_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
    CAUSAL_MASK_VALUE: tl.constexpr,
    a_k_out,  # new argument
):
    m_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)
    bid = batch_head_idx // NUM_HEADS
    hid = batch_head_idx % NUM_HEADS

    qkv_offset = batch_head_idx.to(tl.int64) * stride_head
    Q_base = Q + qkv_offset
    K_base = K + qkv_offset
    V_base = V + qkv_offset
    O_base = O + qkv_offset
    offs_m = m_idx
    mask_m = offs_m < seq_len

    q_ptrs = Q_base + offs_m * stride_seq + tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptrs, mask=mask_m, other=0.0).to(tl.float32)  # shape: [HEAD_DIM]

    theta_ptr = thetas + hid * HEAD_DIM + tl.arange(0, HEAD_DIM)
    theta_this_head = tl.load(theta_ptr).to(tl.float32)
    a_k_ptr = a_k_out + batch_head_idx * seq_len
    prev = 0.0

    for idx in range(seq_len):
        k_ptr = K_base + idx * stride_seq + tl.arange(0, HEAD_DIM)
        k_i = tl.load(k_ptr)
        dot = tl.sum(q * k_i)
        prev = prev + dot
        tl.store(a_k_ptr + idx, prev)  # store directly to global memory

    a_k_q = tl.load(a_k_ptr + offs_m)  # Use the prefix sum for the current position
    freqs_q = a_k_q * theta_this_head  # [HEAD_DIM]
    cos_q = tl.cos(freqs_q)
    sin_q = tl.sin(freqs_q)
    idx = tl.arange(0, HEAD_DIM)
    even_mask = idx % 2 == 0
    odd_mask = idx % 2 == 1

    # --- Q rotary embedding using even/odd mask and swap logic (with tl.load) ---
    idx = tl.arange(0, HEAD_DIM)
    even_mask = (idx % 2 == 0)
    odd_mask = (idx % 2 == 1)
    # q_swap: shift left by 1, pad last with 0
    swap_offsets = idx + 1
    swap_mask = swap_offsets < HEAD_DIM
    q_swap = tl.load(q_ptrs + swap_offsets, mask=swap_mask, other=0.0)
    freqs_cos = cos_q
    freqs_sin = sin_q
    q_rot = (
        even_mask * (freqs_cos * q - freqs_sin * q_swap)
        + odd_mask * (freqs_sin * q_swap + freqs_cos * q)
    )

    o_acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    l_acc = 0.0
    m_acc = -1e9

    for start_n in tl.range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len

        k_ptrs = K_base + offs_n[:, None] * stride_seq + tl.arange(0, HEAD_DIM)[None, :]
        v_ptrs = V_base + offs_n[:, None] * stride_seq + tl.arange(0, HEAD_DIM)[None, :]
        k_block = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v_block = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        a_k_vals = tl.load(a_k_ptr + offs_n)  # [BLOCK_N]
        freqs_k = a_k_vals[:, None] * theta_this_head[None, :]
        cos_k = tl.cos(freqs_k)
        sin_k = tl.sin(freqs_k)
        # --- K rotary embedding using even/odd mask and swap logic (with tl.load) ---
        idx_k = tl.arange(0, HEAD_DIM)
        even_mask_k = (idx_k % 2 == 0)[None, :]
        odd_mask_k = (idx_k % 2 == 1)[None, :]
        swap_offsets_k = idx_k + 1
        swap_mask_k = swap_offsets_k < HEAD_DIM
        k_swap = tl.load(k_ptrs + swap_offsets_k[None, :], mask=mask_n[:, None] & swap_mask_k[None, :], other=0.0)
        freqs_cos_k = cos_k
        freqs_sin_k = sin_k
        k_rot = (
            even_mask_k * (freqs_cos_k * k_block - freqs_sin_k * k_swap)
            + odd_mask_k * (freqs_sin_k * k_swap + freqs_cos_k * k_block)
        )
        logits = tl.sum(q_rot * k_rot, axis=1) * SM_SCALE

        if CAUSAL:
            causal_mask = offs_m >= offs_n
            attn_mask = causal_mask & mask_n
        else:
            attn_mask = mask_n

        logits = tl.where(attn_mask, logits, CAUSAL_MASK_VALUE)
        m_new = tl.maximum(m_acc, tl.max(logits, axis=0))
        alpha = tl.exp(m_acc - m_new)
        p = tl.exp(logits - m_new)
        o_acc = o_acc * alpha + tl.sum(p[:, None] * v_block, axis=0)
        l_acc = l_acc * alpha + tl.sum(p)
        m_acc = m_new

    o_acc = o_acc / l_acc

    o_ptrs = O_base + offs_m * stride_seq + tl.arange(0, HEAD_DIM)
    tl.store(o_ptrs, o_acc.to(O.dtype.element_ty), mask=mask_m)
    m_ptrs = M + batch_head_idx * seq_len + offs_m
    tl.store(m_ptrs, m_acc, mask=mask_m)
    l_ptrs = L + batch_head_idx * seq_len + offs_m
    tl.store(l_ptrs, l_acc, mask=mask_m)
def co_rope_forward(q, k, v, causal, sm_scale, theta=10000.0, a_k_out=None):
    batch_size, num_heads, seq_len, head_dim = q.shape

    assert k.shape == q.shape
    assert v.shape == q.shape
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be CUDA tensors"

    # Precompute thetas
    thetas = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=q.device).float() / head_dim)
    )
    thetas = thetas.repeat_interleave(2)
    thetas = thetas.unsqueeze(0).repeat(num_heads, 1).contiguous()
    
    o = torch.empty_like(q)
    m = torch.empty(
        (batch_size, num_heads, seq_len), device=q.device, dtype=torch.float32
    )
    l = torch.empty(
        (batch_size, num_heads, seq_len), device=q.device, dtype=torch.float32
    )

    if a_k_out is None:
        a_k_out = torch.empty((batch_size, num_heads, seq_len), device=q.device, dtype=torch.float32)

    def grid(meta):
        return (seq_len, batch_size * num_heads)

    co_rope_fwd_kernel[grid](
        Q=q.contiguous(),
        K=k.contiguous(),
        V=v.contiguous(),
        O=o,
        M=m,
        L=l,
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
        a_k_out=a_k_out,
    )
    return (q, k, v, o, m, l, a_k_out), o