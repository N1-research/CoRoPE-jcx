import torch
import triton
import triton.language as tl


@triton.jit
def co_rope_bwd_kernel(
    Q,
    K,
    V,
    O,
    M,
    L,
    A,
    DO,
    DQ,
    DV,  # dv 全局 buffer
    DK,  # 新增 dk 全局 buffer
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    stride_a,
    thetas,
    seq_len,
    SM_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    m_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)

    qkv_offset = batch_head_idx.to(tl.int64) * stride_head

    Q_base = Q + qkv_offset
    K_base = K + qkv_offset
    V_base = V + qkv_offset
    O_base = O + qkv_offset
    DO_base = DO + qkv_offset
    DQ_base = DQ + qkv_offset
    DV_base = DV + qkv_offset
    DK_base = DK + qkv_offset  # 新增 DK_base

    # [D]
    thetas = tl.load(thetas + tl.arange(0, HEAD_DIM)).to(tl.float32)
    swap_offsets = ((tl.arange(0, HEAD_DIM) + 1) % 2) * 2 - 1

    # [D]
    q_ptrs = Q_base + m_idx * stride_seq + tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptrs).to(tl.float32)
    o_ptrs = O_base + m_idx * stride_seq + tl.arange(0, HEAD_DIM)
    o = tl.load(o_ptrs).to(tl.float32)
    do_ptrs = DO_base + m_idx * stride_seq + tl.arange(0, HEAD_DIM)
    do = tl.load(do_ptrs).to(tl.float32)
    m_ptrs = M + batch_head_idx * seq_len + m_idx
    m = tl.load(m_ptrs)
    l_ptrs = L + batch_head_idx * seq_len + m_idx
    l = tl.load(l_ptrs)
    a_ptrs = A + batch_head_idx * stride_a + m_idx
    a_total = tl.load(a_ptrs)
    a_acc = 0.0
    dq_acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for start_n in tl.range(0, m_idx, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n <= m_idx
        k_ptrs = K_base + offs_n[:, None] * stride_seq + tl.arange(0, HEAD_DIM)[None, :]
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
        k_swap = tl.load(k_ptrs + swap_offsets[None, :], mask=mask_n[:, None], other=0.0).to(tl.float32)
        v_ptrs = V_base + offs_n[:, None] * stride_seq + tl.arange(0, HEAD_DIM)[None, :]
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
        z = tl.sigmoid(tl.sum(q * k, axis=1))
        z = tl.where(mask_n, z, 0)
        zc = tl.cumsum(z)
        a = zc - a_total
        freqs = a[:, None] * thetas[None, :]
        freqs_cos = tl.cos(freqs)
        freqs_sin = tl.sin(freqs)
        even_mask = (tl.arange(0, HEAD_DIM) % 2 == 0)[None, :]
        odd_mask = (tl.arange(0, HEAD_DIM) % 2 == 1)[None, :]
        rot_k = (
            even_mask * (freqs_cos * k + freqs_sin * k_swap)
            + odd_mask * (-freqs_sin * k_swap + freqs_cos * k)
        )
        qk = tl.sum(q[None, :] * rot_k, axis=1) * SM_SCALE
        p = tl.exp(qk - m) / l
        dp = tl.sum(do[None, :] * v, axis=1)
        ds = p * (dp - tl.sum(do * o))
        dq_acc += tl.sum(ds[:, None] * rot_k, axis=0) * SM_SCALE
        drot_k = ds[:, None] * q[None, :]
        dr = tl.div_rn(drot_k, tl.where(k != 0, k, float('inf')))
        dr_swap = tl.div_rn(drot_k, tl.where(k_swap != 0, k_swap, float('inf')))
        dr_cos = dr
        dr_sin = dr_swap * tl.where(tl.arange(0, HEAD_DIM) % 2 == 0, -1, 1)
        dfreqs = -tl.sin(dr_cos) + tl.cos(dr_sin)
        da = tl.sum(dfreqs * thetas[None, :], axis=1)
        a_acc = a_acc + tl.sum(da)
        dz = a_acc - tl.cumsum(da, reverse=True)
        dq_acc += tl.sum(dz[:, None] * k, axis=0)
        # dv 累加（向量化 atomic_add）
        dv_ptrs = DV_base + offs_n[:, None] * stride_seq + tl.arange(0, HEAD_DIM)[None, :]
        dv_update = p[:, None] * do[None, :]
        tl.atomic_add(dv_ptrs, dv_update, mask=mask_n[:, None])
        dk_ptrs = DK_base + offs_n[:, None] * stride_seq + tl.arange(0, HEAD_DIM)[None, :]
        dk_update = ds[:, None] * q[None, :]  # q 已经是当前 query 的
        tl.atomic_add(dk_ptrs, dk_update, mask=mask_n[:, None])

    dq_ptrs = DQ_base + m_idx * stride_seq + tl.arange(0, HEAD_DIM)
    tl.store(dq_ptrs, dq_acc.to(DQ.dtype.element_ty))


def co_rope_backward(saved, do, sm_scale=None, theta=10000.0):
    """
    Co-RoPE attention 反向传播函数

    Args:
        saved: 前向传播保存的张量元组
        do: 输出的梯度张量
        sm_scale: 缩放因子，应与前向传播保持一致

    Returns:
        dq, dk, dv: 输入张量 q, k, v 的梯度
    """
    # 从上下文中恢复保存的张量
    q, k, v, o, m, l, a = saved

    batch_size, num_heads, seq_len, head_dim = q.shape

    # 使用传入的 sm_scale，如果没有则使用默认值
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim**0.5)

    # 初始化梯度张量
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    thetas = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=q.device).float() / head_dim)
    )
    thetas = thetas.repeat_interleave(2)

    # 确保张量在 CUDA 上
    assert q.is_cuda and k.is_cuda and v.is_cuda, "输入张量必须是 CUDA 张量"

    # 网格配置
    def grid_q(meta):
        return (triton.cdiv(seq_len, meta["BLOCK_M"]), batch_size * num_heads)

    def grid_kv(meta):
        return (triton.cdiv(seq_len, meta["BLOCK_N"]), batch_size * num_heads)

    # 启动 DQ/DV 计算 kernel
    co_rope_bwd_kernel.run(
        grid=grid_q,
        warmup=False,
        Q=q.contiguous(),
        K=k.contiguous(),
        V=v.contiguous(),
        O=o.contiguous(),
        M=m.contiguous(),
        L=l.contiguous(),
        A=a.contiguous(),
        DO=do.contiguous(),
        DQ=dq.contiguous(),
        DV=dv.contiguous(),
        DK=dk.contiguous(),  # 新增
        stride_batch=q.stride(0),
        stride_head=q.stride(1),
        stride_seq=q.stride(2),
        stride_dim=q.stride(3),
        stride_a=a.stride(1),
        thetas=thetas,
        seq_len=seq_len,
        SM_SCALE=sm_scale,
        BLOCK_M=1,
        BLOCK_N=32,
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        CAUSAL=True,
    )

    # # 启动 DK, DV 计算 kernel
    # co_rope_bwd_kv_kernel[grid_kv](
    #     Q=q.contiguous(),
    #     K=k.contiguous(),
    #     V=v.contiguous(),
    #     O=o.contiguous(),
    #     M=m.contiguous(),
    #     L=l.contiguous(),
    #     DO=do.contiguous(),
    #     DK=dk,
    #     DV=dv,
    #     stride_batch=q.stride(0),
    #     stride_head=q.stride(1),
    #     stride_seq=q.stride(2),
    #     stride_dim=q.stride(3),
    #     seq_len=seq_len,
    #     SM_SCALE=sm_scale,
    #     BLOCK_M=32,
    #     BLOCK_N=32,
    #     NUM_HEADS=num_heads,
    #     HEAD_DIM=head_dim,
    #     CAUSAL=True,
    # )

    return dq, dk, dv