import torch
import torch.nn as nn

class AttnBlock(nn.Module):

    def __init__(self, in_c):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_c)
        self.attn = nn.MultiheadAttention(in_c, num_heads=1, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).transpose(1, 2)  # [B, H*W, C]

        out, _ = self.attn(h, h, h, need_weights=False)  # [B, H*W, C]
        out = out.view(B, H, W, C).permute(0, 3, 1, 2)   # [B, C, H, W]
        out = x + out

        return out