import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, dim, gated=True):
        super().__init__()
        self.gate = nn.Parameter(torch.ones(1, dim)) if gated else 1.0

    def forward(self, x, eps=1e-4):
        """
        Expects input of shape B, N, D
        """
        x = x / (eps + torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)))
        x = self.gate * x
        return x

def build_padding_mask(x, L, context, mask_value=float('inf')):
    B, N, D = x.shape
    padding_mask = torch.full((len(x), context), mask_value)
    for i in range(B):
        padding_mask[i, :L[i]] = 0
        padding_mask[i, L[i]:]
    return padding_mask

def trunc_normal_(shape, mean=0, std=1, upper=2, lower=-2, device="cpu"):
    x = mean + torch.randn(shape, device=device)
    x.clamp_(lower, upper)
    x *= std / x.std(unbiased=False)
    return x

if __name__ == "__main__":
    x = torch.randn(16,4,4)
    print(RMSNorm(4, gated=False)(x))