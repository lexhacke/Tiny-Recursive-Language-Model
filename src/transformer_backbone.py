from torch import nn
import torch
import torch.nn.functional as F
from rope import generate_angles_1d, apply_angles_1d
from utils import RMSNorm
from einops import rearrange

class TransformerBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.prenorm = config['norm_type'] == 'prenorm'
        for _ in range(config['depth']):
            block = nn.Module()
            block.ln1 = RMSNorm(config['dim'])
            block.ln2 = RMSNorm(config['dim'])
            block.attn = Attention(config['context'], config['dim'], n_heads=config['n_heads'])
            block.mlp = MLPGeGLU(config['dim'])
            if self.prenorm:
                block.a_attn = nn.Parameter(torch.tensor(config['residual_alpha']).float(), requires_grad=config['learnable_alpha'])
                block.a_mlp = nn.Parameter(torch.tensor(config['residual_alpha']).float(), requires_grad=config['learnable_alpha'])
            self.blocks.append(block)

    def forward(self, x, attn_mask):
        if self.prenorm:
            for block in self.blocks:
                x = x + block.attn(block.ln1(x), attn_mask) * block.a_attn
                x = x + block.mlp(block.ln2(x)) * block.a_mlp
        else:
            for block in self.blocks:
                x = block.ln1(x + block.attn(x, attn_mask))
                x = block.ln2(x + block.mlp(x))
        return x

class Attention(nn.Module):
    def __init__(self, context_length, emb_dim, causal=True, n_heads=8):
        super().__init__()
        self.causal = causal
        self.context_length = context_length
        self.n_heads = n_heads
        head_dim = emb_dim // n_heads
        self.qkv = nn.Linear(emb_dim, 3*emb_dim, bias=False)
        self.proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.register_buffer("freq", generate_angles_1d(context_length, head_dim), persistent=False)

    def build_attn_mask(self, attn_mask):
        """
        attn_mask = torch.Tensor of shape B, N

        Returns a non-causal attention mask of shape B, N, N
        """
        padding_mask = attn_mask.unsqueeze(-1)
        padding_maskT = rearrange(padding_mask, "B N S -> B S N")
        padding_mask = (padding_mask @ padding_maskT)
        padding_mask = padding_mask > 0

        causal_mask = torch.tril(padding_mask, 0) == 1

        attn_mask = causal_mask * padding_mask
        return attn_mask[:, None, :, :]

    def forward(self, x, attn_mask):
        """
        x = torch.Tensor of shape B, N, D
        attn_mask = torch.Tensor of shape B, N
        """
        B, N, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = rearrange(q, "B N (h D) -> B h N D", h=self.n_heads)
        k = rearrange(k, "B N (h D) -> B h N D", h=self.n_heads)
        v = rearrange(v, "B N (h D) -> B h N D", h=self.n_heads)

        q = apply_angles_1d(q, self.freq)
        k = apply_angles_1d(k, self.freq)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=(~attn_mask.bool())[:, None, None, :], is_causal=True)
        x = rearrange(x, "B h N D -> B N (h D)")
        x = self.proj(x)
        return x

class MLPGeGLU(nn.Module):
    def __init__(self, dim: int, upsample=2, transpose=False):
        """
        dim = embedding dimension
        tokens = number of tokens per embedding
        """
        super().__init__()
        self.transpose = transpose
        self.dim = dim
        self.linearIn = nn.Linear(dim, upsample*dim, bias=False)
        self.gate = nn.Linear(dim, upsample*dim, bias=False)
        self.linearOut = nn.Linear(upsample*dim, dim, bias=False)

    def forward(self, x: torch.Tensor):
        """
        Requires input to be B N D where N=tokens
        Outputs a singleton for x[-1] (z) of shape B 1 D
        Transposes by N, D axis to create a per-feature affine transform
        """
        x = rearrange(x, "B N D -> B D N") if self.transpose else x # batch of token vectors to batch of per-token feature vectors
        x = self.linearOut(F.gelu(self.linearIn(x)) * self.gate(x))
        x = rearrange(x, "B D N -> B N D") if self.transpose else x # recover x,y,z.
        return x

if __name__ == '__main__':
    from transformers import AutoTokenizer

    config = {
        'norm_type':'prenorm',
        "lr":3e-4,
        "dim":192,
        "context":100,
        "vocab_size":262144,
        "n_heads":8,
        "residual_alpha":0,
        'learnable_alpha':True,
        'attention':False,
        'depth':2,
        'device':'cuda'
    }

    token_embedding = nn.Embedding(
        num_embeddings=config['vocab_size'],
        embedding_dim=config['dim'],
        padding_idx=0
    ).cuda()

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
    tokenizer.padding_side = 'right'
    tokenizer.truncation_side = 'right'

    batch = tokenizer(["This is an example forward pass. I hope it works man.",
                       'Another token woohoo'],
                      padding='max_length',
                      truncation=True,
                      max_length=config['context'],
                      return_tensors='pt',
                  ).to('cuda')

    embeddings = token_embedding(batch['input_ids'])
    print(embeddings.shape, batch['attention_mask'].shape)
    block = TransformerBackbone(config).cuda()
    x = block(embeddings, batch['attention_mask'])
    assert x.isnan().sum() == False
    print(x)