from datasets import load_dataset
from transformers import AutoTokenizer


def create_dataset(
    context_size,
    tokenizer_name,
    mode="train",
    dataset_name="DKYoon/SlimPajama-6B",
):
    """
    Stream SlimPajama (or compatible) data and return tokenized samples.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    ds = load_dataset(dataset_name, streaming=True)[mode]

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=context_size,
            return_tensors="pt",
        )

    return ds.map(tokenize, batched=True, remove_columns=["text"])
