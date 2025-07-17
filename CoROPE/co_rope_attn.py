import torch
from CoROPE.co_rope_forward import co_rope_forward
from CoROPE.co_rope_backward import co_rope_backward

class CoRoPEAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal=True, sm_scale=None, theta=10000.0):
        # Save input shapes and device
        batch_size, num_heads, seq_len, head_dim = q.shape
        device = q.device

        # Prepare a_k_out buffer
        a_k_out = torch.empty((batch_size, num_heads, seq_len), device=device, dtype=torch.float32)

        # Call Triton forward kernel
        (q_out, k_out, v_out, o_out, m_out, l_out, a_k_out), output = co_rope_forward(
            q, k, v, causal=causal, sm_scale=sm_scale, theta=theta, a_k_out=a_k_out
        )

        # Save for backward
        ctx.save_for_backward(q_out, k_out, v_out, o_out, m_out, l_out, a_k_out)
        ctx.causal = causal
        ctx.sm_scale = sm_scale
        ctx.theta = theta

        return output

    @staticmethod
    def backward(ctx, grad_output):
        q_out, k_out, v_out, o_out, m_out, l_out, a_k_out = ctx.saved_tensors
        causal = ctx.causal
        sm_scale = ctx.sm_scale
        theta = ctx.theta

        # Recompute thetas
        batch_size, num_heads, seq_len, head_dim = q_out.shape
        device = q_out.device
        thetas = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
        )
        thetas = thetas.repeat_interleave(2)
        thetas = thetas.unsqueeze(0).repeat(num_heads, 1).contiguous()

        # Call Triton backward kernel
        dq, dk, dv, _ = co_rope_backward(
            q_out, k_out, v_out, o_out, m_out, l_out, a_k_out, grad_output, thetas, causal, sm_scale
        )
        return dq, dk, dv, None, None, None  # None for non-tensor args

import torch.nn as nn

class CoRoPEAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, theta=10000.0, causal=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.theta = theta
        self.causal = causal

    def forward(self, x):
        # x: [batch, seq_len, embed_dim]
        B, S, E = x.shape
        H = self.num_heads
        D = self.head_dim

        # Project and reshape
        q = self.q_proj(x).view(B, S, H, D).transpose(1, 2)  # [B, H, S, D]
        k = self.k_proj(x).view(B, S, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, S, H, D).transpose(1, 2)

        sm_scale = 1.0 / (D ** 0.5)

        # Triton-based attention
        attn_out = CoRoPEAttentionFunction.apply(q, k, v, self.causal, sm_scale, self.theta)  # [B, H, S, D]

        # Merge heads and project out
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, E)
        return self.out_proj(attn_out)

