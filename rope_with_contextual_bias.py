import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """
    应用旋转位置编码到query和key
    """
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
    
    # 应用旋转（简化版本，不使用reshape）
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    
    return q_embed, k_embed


def rotate_half(x):
    """
    旋转向量的一半维度
    """
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


def get_rotary_embeddings(seq_len, dim, device, base=10000):
    """
    生成旋转位置编码
    """
    inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
    t = torch.arange(seq_len, device=device).type_as(inv_freq)
    freqs = torch.einsum('i,j->ij', t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


class ContextualBiasAttention(nn.Module):
    """
    带有contextual偏置项的RoPE注意力机制
    """
    def __init__(self, hidden_size, num_heads, max_seq_len=2048, bias_dropout=0.1, 
                 bias_type='learnable', bias_scale=1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_seq_len = max_seq_len
        self.bias_type = bias_type
        self.bias_scale = bias_scale
        
        # 线性变换层
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Contextual偏置项
        if bias_type == 'learnable':
            self.contextual_bias = nn.Parameter(torch.randn(max_seq_len, max_seq_len) * 0.02)
        elif bias_type == 'relative':
            # 相对位置偏置
            self.relative_bias = nn.Parameter(torch.randn(2 * max_seq_len - 1) * 0.02)
        elif bias_type == 'alibi':
            # ALiBi风格的偏置
            self.alibi_slopes = nn.Parameter(torch.randn(num_heads) * 0.02)
        
        self.bias_dropout = nn.Dropout(bias_dropout)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(module.weight)
        
        if self.bias_type == 'learnable':
            # 初始化contextual bias为下三角矩阵（因果注意力）
            with torch.no_grad():
                self.contextual_bias.data = torch.tril(self.contextual_bias.data)
        elif self.bias_type == 'alibi':
            # 初始化ALiBi slopes
            with torch.no_grad():
                self.alibi_slopes.data = torch.randn_like(self.alibi_slopes.data) * 0.02
    
    def _get_contextual_bias(self, seq_len):
        """获取contextual偏置项"""
        if self.bias_type == 'learnable':
            return self.contextual_bias[:seq_len, :seq_len]
        elif self.bias_type == 'relative':
            # 构建相对位置偏置矩阵
            bias = torch.zeros(seq_len, seq_len, device=self.relative_bias.device)
            for i in range(seq_len):
                for j in range(seq_len):
                    relative_pos = i - j
                    bias[i, j] = self.relative_bias[relative_pos + seq_len - 1]
            return bias
        elif self.bias_type == 'alibi':
            # ALiBi偏置
            bias = torch.zeros(seq_len, seq_len, device=self.alibi_slopes.device)
            for i in range(seq_len):
                for j in range(seq_len):
                    if i >= j:  # 因果注意力
                        bias[i, j] = (i - j) * self.bias_scale
            return bias
        else:
            return torch.zeros(seq_len, seq_len, device=next(self.parameters()).device)
    
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
        
        # 应用旋转位置编码
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        
        # 转置为 [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 添加contextual偏置项
        contextual_bias = self._get_contextual_bias(seq_len)
        contextual_bias = contextual_bias.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        scores = scores + contextual_bias * self.bias_scale
        
        # 应用注意力掩码
        if attention_mask is not None:
            # 扩展掩码到注意力头维度
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # 应用causal mask（确保只能看到前面的token）
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        scores = scores.masked_fill(causal_mask.bool(), float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.bias_dropout(attn_weights)
        
        # 计算输出
        attn_output = torch.matmul(attn_weights, v)
        
        # 转置回 [batch_size, seq_len, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        # 重塑为 [batch_size, seq_len, hidden_size]
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        
        # 输出投影
        output = self.o_proj(attn_output)
        
        return output, attn_weights


def test_rope_with_contextual_bias():
    """测试函数"""
    # 参数设置
    batch_size = 2
    seq_len = 128
    hidden_size = 512
    num_heads = 8
    
    print("测试不同类型的contextual bias:")
    
    # 测试learnable bias
    print("\n1. Learnable Bias:")
    model1 = ContextualBiasAttention(hidden_size, num_heads, max_seq_len=seq_len, bias_type='learnable')
    x = torch.randn(batch_size, seq_len, hidden_size)
    output1, attn1 = model1(x)
    print(f"Learnable bias输出形状: {output1.shape}")
    print(f"Learnable bias统计: mean={output1.mean().item():.4f}, std={output1.std().item():.4f}")
    
    # 测试relative bias
    print("\n2. Relative Bias:")
    model2 = ContextualBiasAttention(hidden_size, num_heads, max_seq_len=seq_len, bias_type='relative')
    output2, attn2 = model2(x)
    print(f"Relative bias输出形状: {output2.shape}")
    print(f"Relative bias统计: mean={output2.mean().item():.4f}, std={output2.std().item():.4f}")
    
    # 测试alibi bias
    print("\n3. ALiBi Bias:")
    model3 = ContextualBiasAttention(hidden_size, num_heads, max_seq_len=seq_len, bias_type='alibi')
    output3, attn3 = model3(x)
    print(f"ALiBi bias输出形状: {output3.shape}")
    print(f"ALiBi bias统计: mean={output3.mean().item():.4f}, std={output3.std().item():.4f}")
    
    return model1, model2, model3


if __name__ == "__main__":
    test_rope_with_contextual_bias() 