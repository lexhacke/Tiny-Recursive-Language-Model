from transformers import AutoTokenizer
import json

config = {  
    'norm_type': 'postnorm',
    "pad_idx":50256,
    "lr":1e-4,
    "dim":384,
    "context":512,
    "vocab_size":50295,
    "n_heads":8,
    "residual_alpha":0,
    'learnable_alpha':True,
    'attention':False,
    'depth':2,
    'device':'cpu',
    'tokenizer': 'microsoft/phi-1',
    'n': 2,
    'T': 2,
}

tok = AutoTokenizer.from_pretrained(config['tokenizer'])
eos = tok(tok.eos_token)['input_ids'][0] if tok.eos_token is not None else None
pad = tok(tok.pad_token)['input_ids'][0] if tok.pad_token is not None else None
print(eos, pad)

assert pad == config['pad_idx'] or eos == config['pad_idx'], f"{pad} {eos} not {config['pad_idx']}"
assert config['vocab_size'] == len(tok), f"{config['vocab_size']} != {len(tok)}"

json.dump(config, open("src/config/config.json", "w"), indent=4)