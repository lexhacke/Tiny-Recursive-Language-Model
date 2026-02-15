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
        self.norm_map = {
            'prenorm': 1,
            'postnorm': 2,
            'nGPT': 3,
            'ortho': 4
        }
        self.norm_type = self.norm_map[config['norm_type']]

        if self.norm_type == self.norm_map['nGPT']:
            self.base_scale = 1.0 / (config['dim'] ** 0.5)
            self.alpha_init_value = 0.05
            self.sqk_init_value = 1.0

        rope_scaling = config.get('rope_scaling', 1.0)
        for _ in range(config['depth']):
            block = nn.Module()
            is_nGPT = self.norm_type == self.norm_map['nGPT']
            block.ln1 = RMSNorm(config['dim'], gated=not is_nGPT)
            block.ln2 = RMSNorm(config['dim'], gated=not is_nGPT)
            block.attn = Attention(
                config['context'],
                config['dim'],
                n_heads=config['n_heads'],
                nGPT=is_nGPT,
                rope_scaling=rope_scaling,
            )
            block.mlp = MLPGeGLU(config['dim'], nGPT=is_nGPT)
            if self.norm_type == self.norm_map['prenorm']:
                block.a_attn = nn.Parameter(torch.tensor(config['residual_alpha']).float(), requires_grad=config['learnable_alpha'])
                block.a_mlp = nn.Parameter(torch.tensor(config['residual_alpha']).float(), requires_grad=config['learnable_alpha'])
            elif self.norm_type == self.norm_map['nGPT']:
                block.a_attn = nn.Parameter(self.base_scale * torch.ones(config['dim']))
                block.a_mlp = nn.Parameter(self.base_scale * torch.ones(config['dim']))
            self.blocks.append(block)

    def forward(self, x, attn_mask):
        if self.norm_type == self.norm_map['prenorm']:
            for block in self.blocks:
                x = x + block.attn(block.ln1(x), attn_mask) * block.a_attn
                x = x + block.mlp(block.ln2(x)) * block.a_mlp
        
        elif self.norm_type == self.norm_map['postnorm']:
            for block in self.blocks:
                x = block.ln1(x + block.attn(x, attn_mask))
                x = block.ln2(x + block.mlp(x))
        
        elif self.norm_type == self.norm_map['nGPT']:
            for block in self.blocks:
                # Attention update
                lr = torch.abs(block.a_attn * (self.alpha_init_value / self.base_scale))
                h_attn = block.attn(x, attn_mask)
                x_norm = F.normalize(x, p=2, dim=-1)
                h_attn_norm = F.normalize(h_attn, p=2, dim=-1)
                x = F.normalize(x_norm + lr * (h_attn_norm - x_norm), p=2, dim=-1)

                # MLP update
                lr = torch.abs(block.a_mlp * (self.alpha_init_value / self.base_scale))
                h_mlp = block.mlp(x)
                x_norm = F.normalize(x, p=2, dim=-1)
                h_mlp_norm = F.normalize(h_mlp, p=2, dim=-1)
                x = F.normalize(x_norm + lr * (h_mlp_norm - x_norm), p=2, dim=-1)
        
        elif self.norm_type == self.norm_map['ortho']:
            pass
        
        return x

class Attention(nn.Module):
    def __init__(self, context_length, emb_dim, causal=True, n_heads=8, nGPT=False, rope_scaling=1.0):
        super().__init__()
        self.causal = causal
        self.context_length = context_length
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads
        self.nGPT = nGPT
        self.rope_scaling = rope_scaling
        if nGPT:
            base_scale = 1.0 / (emb_dim ** 0.5)
            self.sqk_init_value = 1.0
            self.sqk_init_scaling = base_scale
            self.sqk = nn.Parameter(base_scale * torch.ones(emb_dim))
        self.qkv = nn.Linear(emb_dim, 3*emb_dim, bias=False)
        self.proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.register_buffer(
            "freq",
            generate_angles_1d(context_length, self.head_dim, scaling_factor=self.rope_scaling),
            persistent=False,
        )

    def forward(self, x, attn_mask):
        """
        x = torch.Tensor of shape B, N, D
        attn_mask = torch.Tensor of shape B, N
        """
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # Reshape to (B, h, N, d) for RoPE - sequence dim must be second-to-last
        q = rearrange(q, "B N (h d) -> B h N d", h=self.n_heads)
        k = rearrange(k, "B N (h d) -> B h N d", h=self.n_heads)
        v = rearrange(v, "B N (h d) -> B h N d", h=self.n_heads)

        # Apply RoPE first (reference applies RoPE before normalization)
        q = apply_angles_1d(q, self.freq)
        k = apply_angles_1d(k, self.freq)

        if self.nGPT:
            # Transpose to (B, N, h, d) for per-head normalization matching reference
            q = rearrange(q, "B h N d -> B N h d")
            k = rearrange(k, "B h N d -> B N h d")
            # sqk scaling: reshape to (1, 1, n_heads, head_dim) for per-head application
            sqk = (self.sqk * (self.sqk_init_value / self.sqk_init_scaling)).view(1, 1, self.n_heads, self.head_dim)
            # L2 normalize per-head (over head_dim), then scale
            q = sqk * F.normalize(q, p=2, dim=-1)
            k = sqk * F.normalize(k, p=2, dim=-1)
            # Transpose back for attention
            q = rearrange(q, "B N h d -> B h N d")
            k = rearrange(k, "B N h d -> B h N d")
            scale = self.head_dim ** 0.5
        else:
            scale = 1.0 / (self.head_dim ** 0.5)

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=scale)
        x = rearrange(x, "B h N d -> B N (h d)")
        x = self.proj(x)
        return x

class MLPGeGLU(nn.Module):
    def __init__(self, dim: int, upsample=2, transpose=False, nGPT=False):
        """
        dim = embedding dimension
        tokens = number of tokens per embedding
        """
        super().__init__()
        self.transpose = transpose
        self.dim = dim
        self.nGPT = nGPT
        self.linearIn = nn.Linear(dim, upsample*dim, bias=False)
        self.gate = nn.Linear(dim, upsample*dim, bias=False)
        self.linearOut = nn.Linear(upsample*dim, dim, bias=False)
        if nGPT:
            self.suv_init_value = 1.0
            self.suv = nn.Parameter(torch.ones(upsample*dim))

    def forward(self, x: torch.Tensor):
        """
        Requires input to be B N D where N=tokens
        Outputs a singleton for x[-1] (z) of shape B 1 D
        Transposes by N, D axis to create a per-feature affine transform
        """
        x = rearrange(x, "B N D -> B D N") if self.transpose else x # batch of token vectors to batch of per-token feature vectors
        h_in = self.linearIn(x)
        h_gate = self.gate(x)
        if self.nGPT:
            suv = self.suv * (self.suv_init_value * (self.dim ** 0.5))
            h_in = suv * h_in
            h_gate = suv * h_gate
        x = self.linearOut(F.gelu(h_in) * h_gate)
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
