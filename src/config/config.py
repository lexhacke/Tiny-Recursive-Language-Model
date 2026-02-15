from transformers import AutoTokenizer
import json

config = {  
    "norm_type": "nGPT",
    "pad_idx": 50256,
    "lr": 0.0002,
    "dim": 768,
    "context": 4096,
    "vocab_size": 50295,
    "n_heads": 8,
    "residual_alpha": 0,
    "learnable_alpha": True,
    "depth": 12,
    "device": "cpu",
    "tokenizer": "microsoft/phi-1",
    "n": 3,
    "T": 6,
    "weight_tying": False,
    "clip_graph": True,
    "threshold": 0.5,
    "exit_early": False,
    "mask_tokens": False,
    "optimizer": "adamw",
    "batch_size": 4,
    "wsd": {
        "warmup_steps": 2000,
        "min_lr_scale": 0.0,
        "final_decay_ratio": 0.10
    }
}

tok = AutoTokenizer.from_pretrained(config['tokenizer'])
eos = tok(tok.eos_token)['input_ids'][0] if tok.eos_token is not None else None
pad = tok(tok.pad_token)['input_ids'][0] if tok.pad_token is not None else None
print(eos, pad)

assert pad == config['pad_idx'] or eos == config['pad_idx'], f"{pad} {eos} not {config['pad_idx']}"
assert config['vocab_size'] == len(tok), f"{config['vocab_size']} != {len(tok)}"

json.dump(config, open("config/config.json", "w"), indent=4)
