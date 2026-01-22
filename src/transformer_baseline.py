from torch import nn
import torch
from utils import RMSNorm
from transformer_backbone import TransformerBackbone
import json

class TransformerBaseline(nn.Module):
    def __init__(self, config):
        """
        'device':'cuda' or 'cpu'
        'vocab_size': int
        'n_heads': int
        'learning_rate': float
        'dim': int
        'output_classes': int
        'depth': int
        'context': int
        """
        super().__init__()
        config['norm_type'] = "postnorm"
        self.dim = config['dim']
        self.context = config['context']
        self.embedding = nn.Embedding(config['vocab_size'], config['dim'])
        self.backbone = TransformerBackbone(config)
        self.final_norm = RMSNorm(config['dim'])
        self.lm_head = nn.Linear(config['dim'], config['vocab_size'], bias=False)
        self.device = config['device']

    def forward(self, x, mask):
        """
        Expects x of shape B, context
        """
        B, _ = x.shape
        x = self.embedding(x)
        x = self.backbone(x, mask)
        x = self.final_norm(x)
        return self.lm_head(x)

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

    slm = TransformerBaseline(config)
    params = sum([p.numel() for p in slm.parameters() if p.requires_grad])
    print(f"Transformer Baseline Parameter Count: {params / 1_000_000:.4}M")
    print(f"Params without lm_head and embedding: {(params - 2 * config['dim'] * config['vocab_size']) / 1_000_000:.4}M")
    pred = slm(batch['input_ids'], batch['attention_mask'])
    assert pred.isnan().sum() == False