from torch import nn, threshold
from torch.nn import functional as F
import torch
from utils import RMSNorm
from transformer_backbone import TransformerBackbone
from utils import trunc_normal_
from einops import rearrange
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
        self.norm_type = config['norm_type']
        self.n = config['n']
        self.T = config['T']
        self.dim = config['dim']
        self.context = config['context']
        self.clip_graph = config['clip_graph']
        self.threshold = config['threshold']
        self.exit_early = config['exit_early']
        self.mask_tokens = config['mask_tokens']

        self.embedding = nn.Embedding(config['vocab_size'], config['dim'])
        self.backbone = TransformerBackbone(config)
        self.lm_head = nn.Linear(config['dim'], config['vocab_size'], bias=False)

        if config['weight_tying']:
            nn.init.normal_(self.embedding.weight, mean=0, std=1/self.dim**0.5)
            self.lm_head.weight = self.embedding.weight # weight tying

        self.conf_head = nn.Linear(config['dim'], 1, bias=False)
        self.norm = RMSNorm(config['dim'])

        if self.norm_type == 'nGPT':
            self.linears = [self.embedding.weight, self.lm_head.weight, self.conf_head.weight]
            for block in self.backbone.blocks:
                self.linears += [block.attn.qkv.weight, 
                                 block.attn.proj.weight, 
                                 block.mlp.linearIn.weight, 
                                 block.mlp.linearOut.weight, 
                                 block.mlp.gate.weight]

        self.device = config['device']

    def inner(self, x, y, z, mask):
        for _ in range(self.n):
            z = self.backbone(x + y + z, mask)
        y = self.backbone(y + z, mask)
        return y, z

    def outer(self, x, y, z, mask):
        preds, conf = [], []
        conf_mask = torch.ones_like(y[:, :, 0:1]) # Shape B, context, 1
        for _ in range(self.T):
            y_next, z_next = self.inner(x, y, z, mask)
            y = y_next * conf_mask + y * (1 - conf_mask)
            z = z_next * conf_mask + z * (1 - conf_mask)
            y_norm = self.norm(y)
            preds.append(self.lm_head(y_norm))
            conf.append(self.conf_head(y_norm))
            proba = F.sigmoid(conf[-1]) # Shape B, context, 1

            if self.mask_tokens:
                conf_mask = conf_mask * (proba < self.threshold).float() # shape B, context, 1
                mask = mask | (conf_mask[:, None, :, :] == 0) # mask queries of confident predictions

            if mask.all() and self.exit_early:
                break

            if self.clip_graph:
                y = y.detach()
                z = z.detach()

        return preds, conf, mask
    
    def forward(self, x, mask, y=None, z=None):
        """
        Expects x of shape B, context

        Returns:
        preds = list[tensor of shape B, context, vocab_size] 
        conf = list[T, B, context, 1]
        """
        B, _ = x.shape
        mask = mask[:, None, :, None].float()
        maskT = rearrange(mask, "B h N S -> B h S N")
        mask = mask @ maskT
        mask = torch.triu(mask)
        mask = mask == 0
        x = self.embedding(x)
        if y is None and z is None:
            y, z = trunc_normal_((2, B, self.context, self.dim),
                                 mean=0,
                                 std=(1/self.context)**0.5,
                                 upper=2,
                                 lower=-2,
                                 device=self.device).chunk(2, dim=0)
            y, z = y[0], z[0]
        preds, conf, mask = self.outer(x, y, z, mask)
        return preds, conf, mask

if __name__ == "__main__":
    from transformers import AutoTokenizer
    import matplotlib.pyplot as plt

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

    pred, conf, mask = slm(batch['input_ids'], batch['attention_mask'])
    plt.imshow(mask[0, 0].detach().cpu().numpy(), cmap='gray')
    plt.show()

    conf, pred = torch.stack(conf), torch.stack(pred)
    print(f"Pred shape: {pred.shape}, Conf shape: {conf.shape}")

    pred_ids = pred.argmax(dim=-1) # Fake token targets
    conf_gt = (pred_ids == batch['input_ids']).float() # Fake confidence targets

    conf_gt = rearrange(conf_gt, "... N -> (...) N")
    conf = rearrange(conf.squeeze(-1), "... N -> (...) N")

    CE = nn.CrossEntropyLoss(ignore_index=config['pad_idx'], reduction='mean')
    BCE = nn.BCEWithLogitsLoss(reduction='mean')
    target = batch['input_ids'].expand(pred.shape[0], -1, -1)
    target = rearrange(target, "... -> (...)")
    pred = rearrange(pred, "... D -> (...) D")
    print("Confidence Cross Entropy Loss:", BCE(conf, conf_gt))
    print("Prediction Cross Entropy Loss:", CE(pred, target))
