from torch import nn
from datasets import load_dataset
from transformers import AutoTokenizer
import torch

def create_dataset(context_size, tokenizer, mode="train", dataset_name="DKYoon/SlimPajama-6B"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Pad token ID:", tokenizer(tokenizer.pad_token)['input_ids'][0])
    tokenizer.padding_side = 'right'
    tokenizer.truncation_side = 'right'
    ds = load_dataset(dataset_name, streaming=True)[mode]

    tokenize = lambda batch : tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=context_size, # might need to optimize this later
        return_tensors="pt"
    )

    ds = ds.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
    )

    return ds

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import json
    from src.trm import TinyRecursiveLM
    from einops import rearrange

    config = json.load(open("config/config.json", "r"))

    cuda_config = {x:config[x] for x in config}
    cuda_config['device'] = 'cuda'

    SLM = TinyRecursiveLM(cuda_config).cuda()
    ds = create_dataset(config['context'], config['tokenizer'])
    dl = DataLoader(
            ds,
            batch_size=16,
            pin_memory=True,
            persistent_workers=True,
            num_workers=4
        )

    CE = nn.CrossEntropyLoss(ignore_index=config['pad_idx'], reduction='mean')
    try:
        for batch in dl:
            max_id = batch['input_ids'].max()
            print(f"Max token ID in batch: {max_id}")
            print(f"Your Model Vocab Size: {config['vocab_size']}")
            if max_id >= config['vocab_size']:
                print(max_id, config['vocab_size'])
                print("You need to increase your vocab size")
            pred, y, z = SLM(batch['input_ids'].cuda(), batch['attention_mask'].cuda())
            print(pred.shape, y.shape, z.shape)

            shifted_logits = pred[..., :-1, :].contiguous()
            shifted_labels = batch['input_ids'].cuda()[..., 1:].contiguous()

            shifted_logits = rearrange(shifted_logits, "B N D -> (B N) D")
            shifted_labels = rearrange(shifted_labels, "B N -> (B N)")

            print(shifted_logits.mean(), shifted_logits.std())

            print("Loss:", CE(shifted_logits, shifted_labels))
            print("Dummy Loss:", CE(torch.randn_like(shifted_logits), shifted_labels))
            break
    finally:
        import gc
        del ds, dl, batch, pred, y, z, shifted_logits, shifted_labels, SLM
        gc.collect()
        torch.cuda.empty_cache()