from datasets import load_dataset, IterableDataset
from transformers import AutoTokenizer
import torch
import hashlib


def create_dataset(
    context_size,
    tokenizer_name,
    mode="train",
    dataset_name="DKYoon/SlimPajama-6B",
    split="train",
    val_fraction=0.0,
    split_seed=0,
):
    """
    Stream SlimPajama (or compatible) data and return tokenized samples.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    base_stream = load_dataset(dataset_name)[mode]

    stride = None
    if val_fraction and val_fraction > 0:
        stride = max(1, int(round(1.0 / val_fraction)))

    def generator():
        for idx, sample in enumerate(base_stream):
            if stride:
                h = hashlib.blake2b(f"{idx}-{split_seed}".encode(), digest_size=4).digest()
                bucket = int.from_bytes(h, "big")
                is_val = (bucket % stride) == 0
                if split == "val" and not is_val:
                    continue
                if split != "val" and is_val:
                    continue
            elif split == "val":
                continue

            encoded = tokenizer(
                sample["text"],
                truncation=True,
                padding="max_length",
                max_length=context_size,
                return_tensors="pt",
            )
            yield {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
            }

    return IterableDataset.from_generator(generator)
