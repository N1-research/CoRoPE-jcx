import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def apply_comrope_pos_emb(q, k, cos, sin, position_ids, contextual_bias=None):
    """
    应用ComRoPE位置编码
    Args:
        q, k: [batch_size, num_heads, seq_len, head_dim]
        cos, sin: [seq_len, head_dim]
        position_ids: [seq_len]
        contextual_bias: [batch_size, num_heads, seq_len, seq_len] (可选)
    """
    # 获取位置编码
    cos = cos[position_ids]  # [seq_len, head_dim]
    sin = sin[position_ids]  # [seq_len, head_dim]
    
    # 扩展维度以匹配q和k的形状
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    
    # 广播到正确的形状
    cos = cos.expand(q.shape[0], q.shape[1], -1, -1)  # [batch_size, num_heads, seq_len, head_dim]
    sin = sin.expand(q.shape[0], q.shape[1], -1, -1)  # [batch_size, num_heads, seq_len, head_dim]
    
    # 应用标准旋转位置编码
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    
    # 如果有contextual bias，添加到旋转后的向量中
    if contextual_bias is not None:
        # contextual_bias shape: [batch_size, num_heads, seq_len, seq_len]
        # 将其应用到q和k上
        q_embed = q_embed + contextual_bias.mean(dim=-1, keepdim=True) * 0.1
        k_embed = k_embed + contextual_bias.mean(dim=-1, keepdim=True) * 0.1
    
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


class ComRoPEAttention(nn.Module):
    """
    ComRoPE注意力机制
    结合了标准RoPE和contextual信息
    """
    
    def __init__(self, hidden_size, num_heads, max_seq_len=2048, 
                 contextual_dropout=0.1, use_contextual_bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_seq_len = max_seq_len
        self.use_contextual_bias = use_contextual_bias
        
        # 线性变换层
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Contextual bias层
        if use_contextual_bias:
            self.contextual_bias = nn.Parameter(torch.randn(max_seq_len, max_seq_len) * 0.02)
            self.contextual_dropout = nn.Dropout(contextual_dropout)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(module.weight)
        
        if self.use_contextual_bias:
            # 初始化contextual bias为下三角矩阵（因果注意力）
            with torch.no_grad():
                self.contextual_bias.data = torch.tril(self.contextual_bias.data)
    
    def _get_contextual_bias(self, batch_size, num_heads, seq_len):
        """获取contextual bias"""
        if not self.use_contextual_bias:
            return None
        
        bias = self.contextual_bias[:seq_len, :seq_len]
        # 扩展为 [batch_size, num_heads, seq_len, seq_len]
        bias = bias.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
        return self.contextual_dropout(bias)
    
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
        
        # 获取contextual bias
        contextual_bias = self._get_contextual_bias(batch_size, self.num_heads, seq_len)
        
        # 应用ComRoPE位置编码
        q, k = apply_comrope_pos_emb(q, k, cos, sin, position_ids, contextual_bias)
        
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


class ComRoPELayer(nn.Module):
    """
    完整的ComRoPE层，包含注意力、残差连接和层归一化
    """
    
    def __init__(self, hidden_size, num_heads, max_seq_len=2048, 
                 dropout=0.1, contextual_dropout=0.1, use_contextual_bias=True):
        super().__init__()
        self.attention = ComRoPEAttention(
            hidden_size, num_heads, max_seq_len, contextual_dropout, use_contextual_bias
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, attention_mask=None, position_ids=None):
        # 注意力层
        attn_output, attn_weights = self.attention(x, attention_mask, position_ids)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x, attn_weights


def test_comrope():
    """测试ComRoPE实现"""
    print("测试ComRoPE实现:")
    
    # 参数设置
    batch_size = 2
    seq_len = 64
    hidden_size = 256
    num_heads = 4
    
    # 创建ComRoPE层
    model = ComRoPELayer(hidden_size, num_heads, max_seq_len=seq_len, use_contextual_bias=True)
    
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


def compare_comrope_with_standard():
    """比较ComRoPE与标准RoPE"""
    print("\n比较ComRoPE与标准RoPE:")
    
    batch_size = 2
    seq_len = 64
    hidden_size = 256
    num_heads = 4
    
    # ComRoPE（带contextual bias）
    model_comrope = ComRoPELayer(hidden_size, num_heads, max_seq_len=seq_len, use_contextual_bias=True)
    
    # 标准RoPE（无contextual bias）
    model_standard = ComRoPELayer(hidden_size, num_heads, max_seq_len=seq_len, use_contextual_bias=False)
    
    x = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, seq_len)
    
    with torch.no_grad():
        output_comrope, attn_comrope = model_comrope(x, attention_mask)
        output_standard, attn_standard = model_standard(x, attention_mask)
        
        print(f"ComRoPE输出统计: mean={output_comrope.mean().item():.4f}, std={output_comrope.std().item():.4f}")
        print(f"标准RoPE输出统计: mean={output_standard.mean().item():.4f}, std={output_standard.std().item():.4f}")
        
        # 计算差异
        output_diff = torch.abs(output_comrope - output_standard)
        attn_diff = torch.abs(attn_comrope - attn_standard)
        
        print(f"输出差异: mean={output_diff.mean().item():.4f}, max={output_diff.max().item():.4f}")
        print(f"注意力权重差异: mean={attn_diff.mean().item():.4f}, max={attn_diff.max().item():.4f}")


if __name__ == "__main__":
    test_comrope()
    compare_comrope_with_standard() 