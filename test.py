import torch
from CoROPE.co_rope_attn import CoRoPEAttention

# Example parameters
embed_dim = 512
num_heads = 8
batch_size = 2
seq_len = 128

# Create the attention module
attn = CoRoPEAttention(embed_dim=512, num_heads=8)
attn = attn.cuda()

# Create dummy input
x = torch.randn(batch_size, seq_len, embed_dim, device='cuda')

# Forward pass
out = attn(x)
print("Output shape:", out.shape)
print("Output dtype:", out.dtype)
print("Any NaN in output?", torch.isnan(out).any().item())
print("Any Inf in output?", torch.isinf(out).any().item())
out.sum().backward()
for name, param in attn.named_parameters():
    print(f"{name} device: {param.device}")
    if param.grad is not None:
        print(f"{name} grad mean: {param.grad.mean().item()}")
    else:
        print(f"{name} grad is None")

print("Input device:", x.device)
