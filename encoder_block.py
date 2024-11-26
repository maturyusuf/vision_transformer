
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST
import multihead_attention
from multihead_attention import MSA

class EncoderBlock(nn.Module):
  def __init__(self, hidden_d, n_heads, mlp_ratio = 4):
    super().__init__()
    self.hidden_d = hidden_d
    self.n_heads = n_heads
    self.norm1 = nn.LayerNorm(normalized_shape=self.hidden_d)
    self.norm2 = nn.LayerNorm(normalized_shape=self.hidden_d)
    self.mhsa = MSA(hidden_d, n_heads)
    self.mlp = nn.Sequential(
        nn.Linear(hidden_d, hidden_d * mlp_ratio),
        nn.GELU(),
        nn.Linear(hidden_d * mlp_ratio, hidden_d)
    )

  def forward(self, x):
    out = x + self.mhsa(self.norm1(x))
    out = out + self.mlp(self.norm2(out))
    return out