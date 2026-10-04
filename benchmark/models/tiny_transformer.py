import math

import torch.nn as nn


# (batch, sequence length, model width)
INPUT_SHAPE = (1, 128, 256)


class Block(nn.Module):
    """A post-norm encoder block, like nn.TransformerEncoderLayer.

    Attention is written out by hand: nn.MultiheadAttention exports through
    scaled_dot_product_attention, which needs opset 14.
    """

    def __init__(self, width, heads, hidden):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.ff = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width))
        # Below opset 17 each of these is exported decomposed into nine nodes.
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)

    def forward(self, x):
        b, n, width = x.shape
        head = width // self.heads
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, head).permute(2, 0, 3, 1, 4)
        attention = (q @ k.transpose(-2, -1) / math.sqrt(head)).softmax(-1)
        x = self.norm1(x + self.proj((attention @ v).transpose(1, 2).reshape(b, n, width)))
        return self.norm2(x + self.ff(x))


def build():
    return nn.Sequential(*(Block(256, heads=4, hidden=1024) for _ in range(4)))
