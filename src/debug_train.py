import torch
import json
print("Step 1: Imports successful")

config = json.load(open("config/config.json", "r"))
print("Step 2: Config loaded")

from trm import TinyRecursiveLM
print("Step 3: Imported TinyRecursiveLM")

cuda_config = {x:config[x] for x in config}
cuda_config['device'] = 'cuda'
print("Step 4: Created cuda_config")

print(f"Creating model with dim={cuda_config['dim']}, context={cuda_config['context']}, vocab={cuda_config['vocab_size']}")
model = TinyRecursiveLM(cuda_config)
print("Step 5: Model created (on CPU)")

print("Moving model to CUDA...")
model = model.to('cuda')
print("Step 6: Model on CUDA")

print("Creating dataset...")
from dataset import create_dataset
ds = create_dataset(cuda_config['context'], cuda_config['tokenizer'])
print("Step 7: Dataset created")

print("Creating DataLoader...")
from torch.utils.data import DataLoader
dl = DataLoader(ds, batch_size=2, num_workers=0)
print("Step 8: DataLoader created")

print("Getting first batch...")
batch = next(iter(dl))
print("Step 9: Got first batch")

print("Running forward pass...")
with torch.no_grad():
    pred, conf, mask = model(batch['input_ids'].to('cuda'), batch['attention_mask'].to('cuda'))
print("Step 10: Forward pass successful!")

print(f"Pred: {len(pred)} tensors, Conf: {len(conf)} tensors")
print("ALL TESTS PASSED!")
