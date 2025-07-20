import torch
import math
import time

# 尝试导入Triton实现
try:
    from CoROPE.co_rope_mqa_forward import co_rope_mqa_forward
    TRITON_AVAILABLE = True
except ImportError:
    print("警告: Triton实现不可用，将只测试PyTorch实现")
    TRITON_AVAILABLE = False


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


def assert_similar(x, y, eps=1e-4, name="tensor", assert_=False, print_=True):
    """正确标准：
    fwd: eps = 1e-4
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


def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.cat([-x2, x1], dim=-1)

def forward_reference_mqa(q, k, v, sm_scale, theta=10000.0):
    batch_size, num_heads, seq_len, head_dim = q.shape
    device = q.device
    o = torch.zeros_like(q)
    thetas = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    thetas = thetas.repeat_interleave(2)
    for b in range(batch_size):
        for h in range(num_heads):
            k_this = k[b, 0]  # [S, D]
            v_this = v[b, 0]  # [S, D]
            m_acc = -1e9
            l_acc = 0.0
            o_acc = torch.zeros(head_dim, device=device)
            a_acc = 0.0
            for t in range(seq_len):
                q_this = q[b, h, t]
                q_swap = rotate_half(q_this)
                mask_n = torch.arange(seq_len, device=device) <= t
                k_valid = k_this[mask_n]
                v_valid = v_this[mask_n]
                z = torch.sigmoid((q_this * k_valid).sum(dim=1))
                zc = torch.cumsum(z, dim=0)
                zs = z.sum()
                a = a_acc + zs - zc
                a_acc = a_acc + zs
                freqs = a.unsqueeze(-1) * thetas.unsqueeze(0)
                freqs_cos = torch.cos(freqs)
                freqs_sin = torch.sin(freqs) * torch.where(
                    torch.arange(head_dim, device=device) % 2 == 0, -1, 1
                )
                rot_q = freqs_cos * q_this + freqs_sin * q_swap
                logits = (rot_q * k_valid).sum(dim=1) * sm_scale
                m_new = max(m_acc, logits.max().item())
                alpha = math.exp(m_acc - m_new)
                p = torch.exp(logits - m_new)
                o_acc = o_acc * alpha + (p.unsqueeze(1) * v_valid).sum(dim=0)
                l_acc = l_acc * alpha + p.sum()
                m_acc = m_new
                o[b, h, t] = o_acc / l_acc
    return o


def test_mqa_corope_comparison():
    print("test_mqa_corope_comparison: Not implemented.")

def test_mqa_vs_standard():
    print("test_mqa_vs_standard: Not implemented.")

def test_corope_features():
    print("test_corope_features: Not implemented.")

if __name__ == "__main__":
    print("测试MQA CoRoPE Triton kernel...")

    # 设置参数
    batch_size = 4
    num_heads = 8
    seq_len = 128
    head_dim = 64

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建随机输入
    torch.manual_seed(42)
    q = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, requires_grad=True
    )
    k = torch.randn(
        batch_size, 1, seq_len, head_dim, device=device, requires_grad=True
    )  # MQA: 共享K
    v = torch.randn(
        batch_size, 1, seq_len, head_dim, device=device, requires_grad=True
    )  # MQA: 共享V

    # Triton实现
    sm_scale = 1.0 / math.sqrt(head_dim)
    
    if TRITON_AVAILABLE and device == "cuda":
        try:
            saved, o_triton = co_rope_mqa_forward(q, k, v, sm_scale=sm_scale)
        except Exception as e:
            print(f"Triton实现出错: {e}")
            o_triton = None
    else:
        print("跳过Triton测试（不可用或不在CUDA设备上）")
        o_triton = None

    # 参考实现
    q_ref = q.clone().detach().requires_grad_(True)
    k_ref = k.clone().detach().requires_grad_(True)
    v_ref = v.clone().detach().requires_grad_(True)
    
    o_ref = forward_reference_mqa(q_ref, k_ref, v_ref, sm_scale=sm_scale)

    print(f"输入形状: q={q.shape}, k={k.shape}, v={v.shape}")
    print(f"PyTorch输出形状: {o_ref.shape}")
    print(f"PyTorch输出统计: mean={o_ref.mean().item():.4f}, std={o_ref.std().item():.4f}")
    
    if o_triton is not None:
        print(f"Triton输出形状: {o_triton.shape}")
        print(f"Triton输出统计: mean={o_triton.mean().item():.4f}, std={o_triton.std().item():.4f}")

    # 正确性检验
    print("\n=== 正确性检验 ===")
    print(f"PyTorch输出包含NaN: {torch.any(torch.isnan(o_ref))}")
    
    if o_triton is not None:
        print(f"Triton输出包含NaN: {torch.any(torch.isnan(o_triton))}")
        assert_similar(o_triton, o_ref, eps=1e-5, name="attention_output")
    else:
        print("跳过Triton比较（不可用）") 

import time

def benchmark_triton_corope_forward(
    B=1, H=2, seq_len=128, D=64, dtype=torch.float16, num_warmup=10, num_reps=50
):
    from CoROPE.co_rope_mqa_forward import co_rope_mqa_forward
    device = "cuda"
    print(f"\nBenchmarking Triton MQA CoRoPE Forward: B={B}, H={H}, S={seq_len}, D={D}, dtype={dtype}")
    q = torch.randn(B, H, seq_len, D, device=device, dtype=dtype)
    k = torch.randn(B, 1, seq_len, D, device=device, dtype=dtype)
    v = torch.randn(B, 1, seq_len, D, device=device, dtype=dtype)
    sm_scale = 1.0 / math.sqrt(D)
    # warmup
    for _ in range(num_warmup):
        _, o = co_rope_mqa_forward(q, k, v, sm_scale=sm_scale)
    torch.cuda.synchronize()
    # timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    stream = torch.cuda.current_stream()
    start_event.record(stream)
    for _ in range(num_reps):
        _, o = co_rope_mqa_forward(q, k, v, sm_scale=sm_scale)
    end_event.record(stream)
    torch.cuda.synchronize()
    ms = start_event.elapsed_time(end_event) / num_reps
    print(f"平均前向耗时: {ms:.3f} ms")
    # FLOPs估算
    flops = B * H * seq_len * seq_len * (4 * D)
    tflops = flops / (ms * 1e-3) / 1e12
    print(f"TFLOP/s: {tflops:.2f}")

if __name__ == "__main__":
    test_mqa_corope_comparison()
    test_mqa_vs_standard()
    test_corope_features()
    if TRITON_AVAILABLE:
        benchmark_triton_corope_forward(B=1, H=2, seq_len=512, D=64, dtype=torch.float16, num_warmup=10, num_reps=50) 