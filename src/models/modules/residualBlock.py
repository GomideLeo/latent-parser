import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.15, batchNorm=False):
        super().__init__()

        self.block = nn.Sequential()

        self.block.append(nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1))
        if batchNorm:
            self.block.append(nn.GroupNorm(min(32, out_c), out_c))
        self.block.append(nn.SiLU())
        if dropout > 0:
            self.block.append(nn.Dropout(dropout))
        self.block.append(nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1))
        if batchNorm:
            self.block.append(nn.GroupNorm(min(32, out_c), out_c))
        
        self.has_shortcut = (in_c != out_c)
        if self.has_shortcut:
            self.shortcut = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = self.block(x)
        if self.has_shortcut:
            x = self.shortcut(x)

        return x + h