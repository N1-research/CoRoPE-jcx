import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """应用旋转位置编码"""
    # 获取位置编码
    cos = cos[position_ids]  # [seq_len, dim]
    sin = sin[position_ids]  # [seq_len, dim]
    
    # 扩展维度以匹配q和k的形状
    # q, k shape: [batch_size, num_heads, seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
    sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
    
    # 广播到正确的形状
    cos = cos.expand(q.shape[0], q.shape[1], -1, -1)  # [batch_size, num_heads, seq_len, dim]
    sin = sin.expand(q.shape[0], q.shape[1], -1, -1)  # [batch_size, num_heads, seq_len, dim]
    
    # 应用旋转（简化版本）
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    
    return q_embed, k_embed


def rotate_half(x):
    """旋转向量的一半维度"""
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


def get_rotary_embeddings(seq_len, dim, device, base=10000):
    """生成旋转位置编码"""
    inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
    t = torch.arange(seq_len, device=device).type_as(inv_freq)
    freqs = torch.einsum('i,j->ij', t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


class SimpleRoPEWithBias(nn.Module):
    """简单的RoPE注意力机制，带有不同的contextual偏置策略"""
    
    def __init__(self, hidden_size, num_heads, max_seq_len=2048, bias_type='learnable'):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_seq_len = max_seq_len
        self.bias_type = bias_type
        
        # 线性变换
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # 根据bias_type初始化不同的偏置
        if bias_type == 'learnable':
            # 可学习的偏置矩阵
            self.bias = nn.Parameter(torch.randn(max_seq_len, max_seq_len) * 0.02)
        elif bias_type == 'alibi':
            # ALiBi风格的偏置
            self.bias_slopes = nn.Parameter(torch.randn(num_heads) * 0.02)
        elif bias_type == 'relative':
            # 相对位置偏置
            self.relative_bias = nn.Parameter(torch.randn(2 * max_seq_len - 1) * 0.02)
        elif bias_type == 'fixed':
            # 固定的偏置（基于距离）
            self.register_buffer('bias', torch.zeros(max_seq_len, max_seq_len))
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(module.weight)
        
        if self.bias_type == 'learnable':
            # 初始化为下三角矩阵（因果注意力）
            with torch.no_grad():
                self.bias.data = torch.tril(self.bias.data)
        elif self.bias_type == 'fixed':
            # 初始化固定偏置
            for i in range(self.max_seq_len):
                for j in range(self.max_seq_len):
                    if i >= j:  # 因果注意力
                        self.bias[i, j] = (i - j) * 0.1
    
    def _get_bias_matrix(self, seq_len):
        """获取偏置矩阵"""
        if self.bias_type == 'learnable':
            return self.bias[:seq_len, :seq_len]
        elif self.bias_type == 'alibi':
            # 构建ALiBi偏置矩阵
            bias = torch.zeros(seq_len, seq_len, device=self.bias_slopes.device)
            for i in range(seq_len):
                for j in range(seq_len):
                    if i >= j:
                        bias[i, j] = (i - j) * 0.1
            return bias
        elif self.bias_type == 'relative':
            # 构建相对位置偏置矩阵
            bias = torch.zeros(seq_len, seq_len, device=self.relative_bias.device)
            for i in range(seq_len):
                for j in range(seq_len):
                    relative_pos = i - j
                    bias[i, j] = self.relative_bias[relative_pos + seq_len - 1]
            return bias
        elif self.bias_type == 'fixed':
            return self.bias[:seq_len, :seq_len]
        else:
            return torch.zeros(seq_len, seq_len, device=next(self.parameters()).device)
    
    def forward(self, x, attention_mask=None):
        """
        Args:
            x: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len] (可选)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # 生成位置ID
        position_ids = torch.arange(seq_len, device=device)
        
        # 生成旋转位置编码
        cos, sin = get_rotary_embeddings(seq_len, self.head_dim, device)
        
        # 线性变换
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 转置为 [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 应用旋转位置编码
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 添加偏置项
        bias_matrix = self._get_bias_matrix(seq_len)
        bias_matrix = bias_matrix.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        scores = scores + bias_matrix
        
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


def test_different_bias_types():
    """测试不同类型的偏置"""
    print("测试不同类型的contextual bias:")
    
    # 参数设置
    batch_size = 2
    seq_len = 64
    hidden_size = 256
    num_heads = 4
    
    bias_types = ['learnable', 'alibi', 'relative', 'fixed']
    
    for bias_type in bias_types:
        print(f"\n{bias_type.upper()} Bias:")
        model = SimpleRoPEWithBias(hidden_size, num_heads, max_seq_len=seq_len, bias_type=bias_type)
        
        x = torch.randn(batch_size, seq_len, hidden_size)
        output, attn_weights = model(x)
        
        print(f"  输出形状: {output.shape}")
        print(f"  输出统计: mean={output.mean().item():.4f}, std={output.std().item():.4f}")
        print(f"  注意力权重统计: mean={attn_weights.mean().item():.4f}, std={attn_weights.std().item():.4f}")
        
        # 检查偏置矩阵
        bias_matrix = model._get_bias_matrix(seq_len)
        print(f"  偏置矩阵统计: mean={bias_matrix.mean().item():.4f}, std={bias_matrix.std().item():.4f}")
        print(f"  偏置矩阵形状: {bias_matrix.shape}")


def compare_with_standard_rope():
    """与标准RoPE比较"""
    print("\n与标准RoPE比较:")
    
    batch_size = 2
    seq_len = 64
    hidden_size = 256
    num_heads = 4
    
    # 标准RoPE（无偏置）
    model_standard = SimpleRoPEWithBias(hidden_size, num_heads, max_seq_len=seq_len, bias_type='none')
    
    # 带偏置的RoPE
    model_bias = SimpleRoPEWithBias(hidden_size, num_heads, max_seq_len=seq_len, bias_type='learnable')
    
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    with torch.no_grad():
        output_standard, attn_standard = model_standard(x)
        output_bias, attn_bias = model_bias(x)
        
        print(f"标准RoPE输出统计: mean={output_standard.mean().item():.4f}, std={output_standard.std().item():.4f}")
        print(f"带偏置RoPE输出统计: mean={output_bias.mean().item():.4f}, std={output_bias.std().item():.4f}")
        
        # 计算差异
        output_diff = torch.abs(output_bias - output_standard)
        attn_diff = torch.abs(attn_bias - attn_standard)
        
        print(f"输出差异: mean={output_diff.mean().item():.4f}, max={output_diff.max().item():.4f}")
        print(f"注意力权重差异: mean={attn_diff.mean().item():.4f}, max={attn_diff.max().item():.4f}")


if __name__ == "__main__":
    test_different_bias_types()
    compare_with_standard_rope() 