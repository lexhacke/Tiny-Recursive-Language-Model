import torch
from torch.utils.checkpoint import checkpoint
class Foo(torch.nn.Module):
    def forward(self, x, mask=None):
        if mask is not None:
            x = x + mask.float()
        return x * 2
foo = Foo()
x = torch.randn(2, requires_grad=True)
mask = torch.ones(2)
print('mask tensor works:', checkpoint(lambda inp, m: foo(inp, m), x, mask))
print('mask None works:', checkpoint(lambda inp, m: foo(inp, m), x, None))
