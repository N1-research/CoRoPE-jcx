
import torch
import triton
import triton.language as tl


@triton.jit
def co_rope_fwd_kernel(
    Q,
    K,
    V,
    O,
    M,
    L,
    A,
    thetas,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    stride_a,
    seq_len,
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

    qkv_offset = batch_head_idx.to(tl.int64) * stride_head
    Q_base = Q + qkv_offset
    K_base = K + qkv_offset
    V_base = V + qkv_offset
    O_base = O + qkv_offset

    offs_m = m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < seq_len

    # [D]
    thetas = tl.load(thetas + tl.arange(0, HEAD_DIM)).to(tl.float32)
    swap_offsets = ((tl.arange(0, HEAD_DIM) + 1) % 2) * 2 - 1  # 1, 0, 3, 2, 5, 4, ...

    # [M, D]
    q_ptrs = Q_base + offs_m[:, None] * stride_seq + tl.arange(0, HEAD_DIM)[None, :]
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    q_swap = tl.load(q_ptrs + swap_offsets[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)

    o_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    l_acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_acc = tl.full([BLOCK_M], -1e9, dtype=tl.float32)
    a_acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    LAST_BLOCK_START = (m_idx // BLOCK_N) * BLOCK_N
    for start_n in tl.range(LAST_BLOCK_START, -1, -BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # [N]
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n <= m_idx

        # [N, D]
        k_ptrs = K_base + offs_n[:, None] * stride_seq + tl.arange(0, HEAD_DIM)[None, :]
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        # [N]
        z = tl.sigmoid(tl.sum(q * k, axis=1))
        z = tl.where(offs_n <= m_idx, z, 0)
        # [N]
        zc = tl.cumsum(z)
        zs = tl.sum(z)
        a = a_acc + zs - zc

        # [M]
        a_acc = a_acc + zs # a[0, :]

        # [N, D] = [N, 1] [1, D]
        freqs = a[:, None] * thetas[None, :]
        freqs_cos = tl.cos(freqs)
        freqs_sin = tl.sin(freqs) * tl.where(tl.arange(0, HEAD_DIM) % 2 == 0, -1, 1)

        # [N, D]
        rot_q = freqs_cos * q + freqs_sin * q_swap

        # [N]
        logits = tl.where(offs_n <= m_idx, tl.sum(rot_q * k, axis=1) * SM_SCALE, CAUSAL_MASK_VALUE)

        # [N, D]
        v_ptrs = V_base + offs_n[:, None] * stride_seq + tl.arange(0, HEAD_DIM)[None, :]
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        # [M]
        m_new = tl.maximum(m_acc, tl.max(logits))
        # [M]
        alpha = tl.exp(m_acc - m_new)
        # [N]
        p = tl.exp(logits - m_new)

        # [M, D]
        o_acc = o_acc * alpha[:, None] + tl.sum(p[:, None] * v, axis=0)
        l_acc = l_acc * alpha + tl.sum(p)
        m_acc = m_new

    # [M, D]
    o_acc = o_acc / l_acc[:, None]

    o_ptrs = O_base + offs_m[:, None] * stride_seq + tl.arange(0, HEAD_DIM)[None, :]
    tl.store(o_ptrs, o_acc.to(O.dtype.element_ty), mask=mask_m[:, None])

    m_ptrs = M + batch_head_idx * seq_len + offs_m
    tl.store(m_ptrs, m_acc, mask=mask_m)

    l_ptrs = L + batch_head_idx * seq_len + offs_m
    tl.store(l_ptrs, l_acc, mask=mask_m)

    a_ptrs = A + batch_head_idx.to(tl.int64) * stride_a + m_idx
    tl.store(a_ptrs[None], a_acc)


def co_rope_forward(q, k, v, sm_scale, causal=True, theta=10000.0):
    batch_size, num_heads, seq_len, head_dim = q.shape

    assert k.shape == q.shape
    assert v.shape == q.shape
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be CUDA tensors"

    thetas = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=q.device).float() / head_dim)
    )
    thetas = thetas.repeat_interleave(2)

    o = torch.empty_like(q)
    m = torch.empty(
        (batch_size, num_heads, seq_len), device=q.device, dtype=torch.float32
    )
    l = torch.empty(
        (batch_size, num_heads, seq_len), device=q.device, dtype=torch.float32
    )
    a = torch.empty(
        (batch_size, num_heads, seq_len), device=q.device, dtype=torch.float32
    )

    def grid(meta):
        return (triton.cdiv(seq_len, meta["BLOCK_M"]), batch_size * num_heads)

    co_rope_fwd_kernel[grid](
        Q=q.contiguous(),
        K=k.contiguous(),
        V=v.contiguous(),
        O=o,
        M=m,
        L=l,
        A=a,
        thetas=thetas,
        stride_batch=q.stride(0),
        stride_head=q.stride(1),
        stride_seq=q.stride(2),
        stride_dim=q.stride(3),
        stride_a=a.stride(1),
        seq_len=seq_len,
        SM_SCALE=sm_scale,
        BLOCK_M=1,
        BLOCK_N=32,
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        CAUSAL=causal,
        CAUSAL_MASK_VALUE=-torch.finfo(torch.float32).max,
    )
    # if os.getenv('COALIBI_VERBOSE', '0') == '1':
    #     try:
    #         best_cfg = _co_alibi_fwd_kernel.get_best_config()
    #         print(f"[Co-ALIBI] Original kernel best config: {best_cfg.kwargs}, num_warps={best_cfg.num_warps}, num_stages={best_cfg.num_stages}")
    #     except Exception as _e:
    #         pass
    return (q, k, v, o, m, l, a), o
