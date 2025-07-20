import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


def skew_symmetric(v):
    """
    将3D向量转换为斜对称矩阵
    Args:
        v: [batch_size, 3] 或 [3]
    Returns:
        skew_matrix: [batch_size, 3, 3] 或 [3, 3]
    """
    if v.dim() == 1:
        v = v.unsqueeze(0)
    
    batch_size = v.shape[0]
    skew = torch.zeros(batch_size, 3, 3, device=v.device)
    
    skew[:, 0, 1] = -v[:, 2]
    skew[:, 0, 2] = v[:, 1]
    skew[:, 1, 0] = v[:, 2]
    skew[:, 1, 2] = -v[:, 0]
    skew[:, 2, 0] = -v[:, 1]
    skew[:, 2, 1] = v[:, 0]
    
    return skew


def so3_exp_map(omega):
    """
    SO(3)指数映射：将李代数映射到李群
    Args:
        omega: [batch_size, 3] 角速度向量
    Returns:
        R: [batch_size, 3, 3] 旋转矩阵
    """
    batch_size = omega.shape[0]
    theta = torch.norm(omega, dim=1, keepdim=True)
    
    # 避免除零
    eps = 1e-8
    theta = torch.clamp(theta, min=eps)
    
    # 归一化角速度
    omega_normalized = omega / theta
    
    # 斜对称矩阵
    omega_skew = skew_symmetric(omega_normalized)
    
    # Rodrigues公式
    I = torch.eye(3, device=omega.device).unsqueeze(0).expand(batch_size, -1, -1)
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    
    R = I + sin_theta * omega_skew + (1 - cos_theta) * torch.matmul(omega_skew, omega_skew)
    
    return R


def so3_log_map(R):
    """
    SO(3)对数映射：将李群映射到李代数
    Args:
        R: [batch_size, 3, 3] 旋转矩阵
    Returns:
        omega: [batch_size, 3] 角速度向量
    """
    batch_size = R.shape[0]
    
    # 计算旋转角度
    trace = torch.diagonal(R, dim1=1, dim2=2).sum(dim=1)
    cos_theta = (trace - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1, 1)
    theta = torch.acos(cos_theta)
    
    # 避免除零
    eps = 1e-8
    sin_theta = torch.sin(theta)
    sin_theta = torch.clamp(sin_theta, min=eps)
    
    # 计算旋转轴
    omega_skew = (R - R.transpose(1, 2)) / (2 * sin_theta.unsqueeze(1).unsqueeze(2))
    
    # 提取角速度向量
    omega = torch.zeros(batch_size, 3, device=R.device)
    omega[:, 0] = omega_skew[:, 2, 1]
    omega[:, 1] = omega_skew[:, 0, 2]
    omega[:, 2] = omega_skew[:, 1, 0]
    
    # 处理特殊情况（theta接近0）
    small_angle_mask = theta < eps
    if small_angle_mask.any():
        omega[small_angle_mask] = 0
    
    return omega * theta.unsqueeze(1)


def se3_exp_map(xi):
    """
    SE(3)指数映射：将李代数映射到李群
    Args:
        xi: [batch_size, 6] 李代数元素 [omega, v]
    Returns:
        T: [batch_size, 4, 4] 变换矩阵
    """
    batch_size = xi.shape[0]
    omega = xi[:, :3]  # 角速度
    v = xi[:, 3:]      # 线速度
    
    # SO(3)部分
    R = so3_exp_map(omega)
    
    # 计算平移部分
    theta = torch.norm(omega, dim=1, keepdim=True)
    eps = 1e-8
    theta = torch.clamp(theta, min=eps)
    
    omega_normalized = omega / theta
    omega_skew = skew_symmetric(omega_normalized)
    
    # 计算V矩阵
    I = torch.eye(3, device=xi.device).unsqueeze(0).expand(batch_size, -1, -1)
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    
    V = I + (1 - cos_theta) / (theta ** 2) * omega_skew + (theta - sin_theta) / (theta ** 3) * torch.matmul(omega_skew, omega_skew)
    
    t = torch.matmul(V, v.unsqueeze(2)).squeeze(2)
    
    # 构建变换矩阵
    T = torch.zeros(batch_size, 4, 4, device=xi.device)
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    T[:, 3, 3] = 1
    
    return T


def se3_log_map(T):
    """
    SE(3)对数映射：将李群映射到李代数
    Args:
        T: [batch_size, 4, 4] 变换矩阵
    Returns:
        xi: [batch_size, 6] 李代数元素
    """
    batch_size = T.shape[0]
    R = T[:, :3, :3]
    t = T[:, :3, 3]
    
    # SO(3)部分
    omega = so3_log_map(R)
    
    # 计算平移部分
    theta = torch.norm(omega, dim=1, keepdim=True)
    eps = 1e-8
    theta = torch.clamp(theta, min=eps)
    
    omega_normalized = omega / theta
    omega_skew = skew_symmetric(omega_normalized)
    
    # 计算V的逆
    I = torch.eye(3, device=T.device).unsqueeze(0).expand(batch_size, -1, -1)
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    
    V_inv = I - 0.5 * omega_skew + (1 - theta * cos_theta / (2 * sin_theta)) / (theta ** 2) * torch.matmul(omega_skew, omega_skew)
    
    v = torch.matmul(V_inv, t.unsqueeze(2)).squeeze(2)
    
    return torch.cat([omega, v], dim=1)


class LieRotationAttention(nn.Module):
    """
    基于Lie rotation的注意力机制
    """
    
    def __init__(self, hidden_size, num_heads, max_seq_len=2048):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_seq_len = max_seq_len
        
        # 线性变换层
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Lie rotation参数
        self.rotation_params = nn.Parameter(torch.randn(max_seq_len, 6) * 0.02)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(module.weight)
    
    def apply_lie_rotation(self, x, position_ids):
        """
        应用Lie rotation到输入
        Args:
            x: [batch_size, seq_len, hidden_size]
            position_ids: [seq_len]
        Returns:
            rotated_x: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # 获取旋转参数
        rotation_params = self.rotation_params[position_ids]  # [seq_len, 6]
        
        # 将hidden_size分成3D向量组
        num_vectors = self.hidden_size // 3
        if self.hidden_size % 3 != 0:
            # 如果不能整除3，用零填充
            padding = 3 - (self.hidden_size % 3)
            x = F.pad(x, (0, padding))
            num_vectors = (self.hidden_size + padding) // 3
        
        # 重塑为3D向量
        x_3d = x.view(batch_size, seq_len, num_vectors, 3)  # [batch_size, seq_len, num_vectors, 3]
        
        # 应用SE(3)变换
        rotated_x_3d = torch.zeros_like(x_3d)
        for i in range(seq_len):
            T = se3_exp_map(rotation_params[i].unsqueeze(0))  # [1, 4, 4]
            T = T.expand(batch_size, -1, -1)  # [batch_size, 4, 4]
            
            # 应用变换
            x_homogeneous = F.pad(x_3d[:, i], (0, 1), value=1)  # [batch_size, num_vectors, 4]
            rotated_homogeneous = torch.matmul(T, x_homogeneous.transpose(1, 2)).transpose(1, 2)
            rotated_x_3d[:, i] = rotated_homogeneous[:, :, :3]
        
        # 重塑回原始形状
        rotated_x = rotated_x_3d.view(batch_size, seq_len, -1)
        if self.hidden_size % 3 != 0:
            rotated_x = rotated_x[:, :, :self.hidden_size]
        
        return rotated_x
    
    def forward(self, x, attention_mask=None, position_ids=None):
        """
        Args:
            x: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len] (可选)
            position_ids: [seq_len] (可选)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # 生成位置ID
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device)
        
        # 应用Lie rotation
        x_rotated = self.apply_lie_rotation(x, position_ids)
        
        # 线性变换
        q = self.q_proj(x_rotated).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x_rotated).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x_rotated).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 转置为 [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用注意力掩码
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # 应用causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        scores = scores.masked_fill(causal_mask.bool(), float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # 计算输出
        attn_output = torch.matmul(attn_weights, v)
        
        # 转置回 [batch_size, seq_len, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        # 重塑为 [batch_size, seq_len, hidden_size]
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        
        # 输出投影
        output = self.o_proj(attn_output)
        
        return output, attn_weights


def test_lie_rotation():
    """测试Lie rotation实现"""
    print("测试Lie rotation实现:")
    
    # 参数设置
    batch_size = 2
    seq_len = 64
    hidden_size = 300  # 确保能被3整除
    num_heads = 4
    
    # 创建Lie rotation注意力层
    model = LieRotationAttention(hidden_size, num_heads, max_seq_len=seq_len)
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, seq_len)
    
    # 前向传播
    output, attn_weights = model(x, attention_mask)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    print(f"输出统计: mean={output.mean().item():.4f}, std={output.std().item():.4f}")
    print(f"注意力权重统计: mean={attn_weights.mean().item():.4f}, std={attn_weights.std().item():.4f}")
    
    return model, output, attn_weights


def test_so3_operations():
    """测试SO(3)操作"""
    print("\n测试SO(3)操作:")
    
    # 创建测试数据
    batch_size = 5
    omega = torch.randn(batch_size, 3) * 0.1
    
    # 指数映射
    R = so3_exp_map(omega)
    print(f"旋转矩阵形状: {R.shape}")
    print(f"旋转矩阵行列式: {torch.det(R)}")
    
    # 对数映射
    omega_recovered = so3_log_map(R)
    print(f"恢复的角速度形状: {omega_recovered.shape}")
    
    # 检查误差
    error = torch.norm(omega - omega_recovered, dim=1)
    print(f"SO(3)映射误差: mean={error.mean().item():.6f}, max={error.max().item():.6f}")


def test_se3_operations():
    """测试SE(3)操作"""
    print("\n测试SE(3)操作:")
    
    # 创建测试数据
    batch_size = 5
    xi = torch.randn(batch_size, 6) * 0.1
    
    # 指数映射
    T = se3_exp_map(xi)
    print(f"变换矩阵形状: {T.shape}")
    print(f"变换矩阵行列式: {torch.det(T[:, :3, :3])}")
    
    # 对数映射
    xi_recovered = se3_log_map(T)
    print(f"恢复的李代数形状: {xi_recovered.shape}")
    
    # 检查误差
    error = torch.norm(xi - xi_recovered, dim=1)
    print(f"SE(3)映射误差: mean={error.mean().item():.6f}, max={error.max().item():.6f}")


if __name__ == "__main__":
    test_so3_operations()
    test_se3_operations()
    test_lie_rotation() 