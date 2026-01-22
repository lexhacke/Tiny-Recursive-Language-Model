from torch import nn
import torch
from utils import RMSNorm
from transformer_backbone import TransformerBackbone
from utils import trunc_normal_
import json

class TinyRecursiveLM(nn.Module):
    def __init__(self, config):
        """
        'device':'cuda' or 'cpu'
        'vocab_size': int
        'n_heads': int
        'learning_rate': float
        'dim': int
        'output_classes': int
        'n': int
        'T': int
        'depth': int
        'clip_graph': bool
        'context': int
        """
        super().__init__()
        self.n = config['n']
        self.T = config['T']
        self.dim = config['dim']
        self.context = config['context']
        self.embedding = nn.Embedding(config['vocab_size'], config['dim'])
        self.backbone = TransformerBackbone(config)
        self.lm_head = nn.Linear(config['dim'], config['vocab_size'], bias=False)
        self.embedding.weight = nn.Parameter(torch.randn_like(self.embedding.weight.T) / self.dim**0.5)
        self.lm_head.weight = self.embedding.weight # weight tying
        self.norm = RMSNorm(config['dim'])
        self.device = config['device']

    def inner(self, x, y, z, mask, n=6):
        for _ in range(n):
            z = self.backbone(x + y + z, mask)
        y = self.backbone(y + z, mask)
        return y, z

    def outer(self, x, y, z, mask, n=6, T=3, clip_graph=False):
        if clip_graph:
            with torch.no_grad():
                for j in range(T-1):
                    y, z = self.inner(x, y, z, mask, n=n)
        else:
            for j in range(T-1):
                y, z = self.inner(x, y, z, mask, n=n)
        y, z = self.inner(x, y, z, mask, n=n)
        return y, z

    def forward(self, x, mask, y=None, z=None, clip_graph=False):
        """
        Expects x of shape B, context
        """
        B, _ = x.shape
        x = self.embedding(x)
        if y is None and z is None:
            y, z = trunc_normal_((2, B, self.context, self.dim),
                                 mean=0,
                                 std=(1/self.context)**0.5,
                                 upper=2,
                                 lower=-2,
                                 device=self.device).chunk(2, dim=0)
            y, z = y[0], z[0]

        y, z = self.outer(x, y, z, mask, n=self.n, T=self.T, clip_graph=clip_graph)
        return self.lm_head(self.norm(y)), y.detach(), z.detach()

if __name__ == "__main__":
    from transformers import AutoTokenizer

    config = json.load(open("config/config.json", "r"))

    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    tokenizer.truncation_side = 'right'
    
    batch = tokenizer(["This is an example forward pass. I hope it works man.",
                       "Example 2. Short af."],
                      padding='max_length',
                      truncation=True,
                      max_length=config['context'],
                      return_tensors='pt'
                  )

    slm = TinyRecursiveLM(config)
    params = sum([p.numel() for p in slm.parameters() if p.requires_grad])
    print(f"TinyRecursiveLM Baseline Parameter Count: {params / 1_000_000:.4}M")
    print(f"Params without lm_head and embedding: {(params - 2 * config['dim'] * config['vocab_size']) / 1_000_000:.4}M")
    pred, y, z = slm(batch['input_ids'], batch['attention_mask'])
    assert pred.isnan().sum() == False