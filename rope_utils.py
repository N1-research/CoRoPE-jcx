import torch


def precompute_rope_freqs(seq_len, head_dim, device="cuda", theta=10000.0):
    """
    预计算RoPE的频率表

    Args:
        seq_len: 序列长度
        head_dim: 头维度
        device: 设备
        theta: 基础频率，通常为10000.0

    Returns:
        cos, sin: 预计算的cos和sin表，形状为(seq_len, head_dim)
    """
    # 确保head_dim是偶数
    assert head_dim % 2 == 0, "head_dim必须是偶数"

    # 计算频率
    freqs = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )

    # 计算位置索引
    t = torch.arange(seq_len, device=device).float()

    # 计算外积得到所有位置和频率的组合
    freqs = torch.outer(t, freqs)  # (seq_len, head_dim//2)

    # 扩展频率以匹配完整的head_dim
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)

    # 创建完整的cos和sin表
    cos = torch.zeros(seq_len, head_dim, device=device)
    sin = torch.zeros(seq_len, head_dim, device=device)

    # 填充cos和sin表
    cos[:, 0::2] = freqs_cos
    cos[:, 1::2] = freqs_cos
    sin[:, 0::2] = freqs_sin
    sin[:, 1::2] = freqs_sin

    return cos, sin


def apply_rope_torch(x, cos, sin):
    """
    使用PyTorch应用RoPE变换（用于参考实现）

    Args:
        x: 输入张量，形状为(batch, heads, seq_len, head_dim)
        cos: cos表，形状为(seq_len, head_dim)
        sin: sin表，形状为(seq_len, head_dim)

    Returns:
        应用RoPE后的张量
    """
    # 获取维度
    batch_size, num_heads, seq_len, head_dim = x.shape

    # 扩展cos和sin以匹配输入维度
    cos = cos.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, seq_len, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, seq_len, head_dim)

    # 分离奇偶维度
    x1 = x[..., 0::2]  # 偶数维度
    x2 = x[..., 1::2]  # 奇数维度

    # 应用旋转变换
    cos_part = cos[..., 0::2]
    sin_part = sin[..., 0::2]

    # 计算旋转后的值
    x1_rot = x1 * cos_part - x2 * sin_part
    x2_rot = x1 * sin_part + x2 * cos_part

    # 重新组合
    x_rot = torch.zeros_like(x)
    x_rot[..., 0::2] = x1_rot
    x_rot[..., 1::2] = x2_rot

    return x_rot


def apply_rope_inverse_torch(x, cos, sin):
    """
    应用RoPE的逆变换（用于梯度反向传播）

    Args:
        x: 输入张量，形状为(batch, heads, seq_len, head_dim)
        cos: cos表，形状为(seq_len, head_dim)
        sin: sin表，形状为(seq_len, head_dim)

    Returns:
        应用RoPE逆变换后的张量
    """
    # 获取维度
    batch_size, num_heads, seq_len, head_dim = x.shape

    # 扩展cos和sin以匹配输入维度
    cos = cos.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, seq_len, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, seq_len, head_dim)

    # 分离奇偶维度
    x1 = x[..., 0::2]  # 偶数维度
    x2 = x[..., 1::2]  # 奇数维度

    # 应用逆旋转变换
    cos_part = cos[..., 0::2]
    sin_part = sin[..., 0::2]

    # 计算逆旋转后的值（注意sin的符号相反）
    x1_rot = x1 * cos_part + x2 * sin_part
    x2_rot = -x1 * sin_part + x2 * cos_part

    # 重新组合
    x_rot = torch.zeros_like(x)
    x_rot[..., 0::2] = x1_rot
    x_rot[..., 1::2] = x2_rot

    return x_rot