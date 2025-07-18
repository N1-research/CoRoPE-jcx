import torch
import math

from CoROPE.forward import co_rope_forward
from CoROPE.co_rope_backward import co_rope_backward
from rope_utils import precompute_rope_freqs, apply_rope_torch

import torch
import math

# import warnings
# warnings.filterwarnings('error', category=RuntimeWarning)

def print_red_warning(message):
    print(f"\033[31mERROR: {message}\033[0m")


def calc_sim(x, y, name="tensor"):
    x, y = x.data.double(), y.data.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        print_red_warning(f"{name} all zero")
        return 1
    sim = 2 * (x * y).sum() / denominator
    return sim


def assert_similar(x, y, eps=1e-8, name="tensor", assert_=False, print_=True):
    """正确标准：
    fwd: eps = 1e-5
    bwd (gradient): eps = 1e-4
    """
    sim = calc_sim(x, y, name)
    diff = 1.0 - sim
    if not (0 <= diff <= eps):
        print_red_warning(f"{name} Error: {diff}")
        if assert_:
            assert False
    else:
        if print_:
            print(f"passed: {name} diff={diff}")


def forward_reference(q, k, v, sm_scale, theta=10000.0):
    batch_num, head_num, seq_len, head_dim = q.shape

    # [B, H, T, T]
    qk = torch.matmul(q, k.transpose(-2, -1))
    # [B, H, T, T]
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=q.device), diagonal=1
    ).bool().expand([batch_num, head_num, seq_len, seq_len])
    # [B, H, T, T]
    z = torch.sigmoid(qk)
    # [B, H, T, T]
    z = z.masked_fill(causal_mask, 0)
    # [B, H, T, T]
    a = torch.cumsum(z, dim=-1)

    # [D]
    thetas: torch.Tensor = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=q.device).float() / head_dim)
    )

    def apply_rotary(d: torch.Tensor, a: torch.Tensor):
        assert len(d.shape) == len(a.shape) + 1
        if len(a.shape) == 0:
            freq = a * thetas
        else:
            freq = torch.outer(a, thetas)
        freq_cos = torch.cos(freq)
        freq_sin = torch.sin(freq)
        
        d_even = d[..., ::2]
        d_odd = d[..., 1::2]
        
        out_even = d_even * freq_cos - d_odd * freq_sin
        out_odd = d_even * freq_sin + d_odd * freq_cos
        
        out = torch.stack((out_even, out_odd), dim=-1).reshape(d.shape)
        return out

    # [B, H, T, T]
    w = torch.zeros([q.shape[0], q.shape[1], seq_len, seq_len], device=q.device)

    for b in range(batch_num):
        for h in range(head_num):
            for t in range(seq_len):
                # [D]
                q_rot = apply_rotary(q[b, h, t, :], a[b, h, t, t]).unsqueeze(0)
                # [T, D]
                k_rot = apply_rotary(k[b, h, :, :], a[b, h, t, :])
                # [T]
                causal_mask = torch.arange(0, seq_len, device=q.device) <= t
                logits = torch.where(causal_mask, torch.sum(k_rot * q_rot, dim=-1) * sm_scale, float('-inf'))

                # [T]
                w[b, h, t, :] = torch.softmax(logits, dim=0)

    # [B, H, T, D]
    o = torch.matmul(w, v)

    return o, w

if __name__ == "__main__":
    print("测试Fused RoPE + Attention Triton kernel...")

    # 设置参数
    batch_size = 1
    num_heads = 1
    seq_len = 4
    head_dim = 4

    device = "cuda"

    # 创建随机输入
    torch.manual_seed(42)
    q = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, requires_grad=True
    )
    k = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, requires_grad=True
    )
    v = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, requires_grad=True
    )
    do = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, requires_grad=True
    )

    # Triton实现
    sm_scale = 1.0 / math.sqrt(head_dim)
    saved, o_triton = co_rope_forward(q, k, v, sm_scale=sm_scale)
    dq, dk, dv = co_rope_backward(saved, do, sm_scale)

    # 参考实现 - 使用torch自动梯度
    # 需要重新创建输入张量以确保梯度计算正确
    q_ref = q.clone().detach().requires_grad_(True)
    k_ref = k.clone().detach().requires_grad_(True)
    v_ref = v.clone().detach().requires_grad_(True)
    
    o_ref, attn_weights = forward_reference(q_ref, k_ref, v_ref, sm_scale=sm_scale)
    
    # 使用torch自动梯度计算反向传播
    o_ref.backward(do)
    dq_ref = q_ref.grad
    dk_ref = k_ref.grad
    dv_ref = v_ref.grad

    print(dq)
    print(dq_ref)

    # 正确性检验
    print("\n=== 正确性检验 ===")
    print(torch.any(torch.isnan(o_triton)), torch.any(torch.isnan(o_ref)))
    assert_similar(o_triton, o_ref, eps=1e-5, name="attention_output")
    assert_similar(dq, dq_ref, eps=1e-5, name="dq")
    assert_similar(dk, dk_ref, eps=1e-5, name="dk")
    assert_similar(dv, dv_ref, eps=1e-5, name="dv")
