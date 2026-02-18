from transformers import AutoTokenizer
import json

config = {
    "norm_type": "nGPT",
    "pad_idx": 50256,
    "lr": 0.0005,
    "dim": 768,
    "context": 1024,
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
    "activation_checkpointing": False,
    "tokens_per_param": 20,
    "val_fraction": 0.001,
    "split_seed": 0,
    "val_check_interval": 100,
    "checkpoint_every_batches": 100,
    "wsd": {
        "warmup_steps": 2000,
        "min_lr_scale": 0.0,
        "final_decay_ratio": 0.1
    }
}

tok = AutoTokenizer.from_pretrained(config['tokenizer'])
eos = tok(tok.eos_token)['input_ids'][0] if tok.eos_token is not None else None
pad = tok(tok.pad_token)['input_ids'][0] if tok.pad_token is not None else None
print(eos, pad)

assert pad == config['pad_idx'] or eos == config['pad_idx'], f"{pad} {eos} not {config['pad_idx']}"
assert config['vocab_size'] == len(tok), f"{config['vocab_size']} != {len(tok)}"

json.dump(config, open("src/config/config.json", "w"), indent=4)
