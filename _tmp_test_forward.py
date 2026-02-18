import json, torch, sys
sys.path.append('src')
from trm import TinyRecursiveLM
from train import LLMLightning

config = json.load(open('src/config/config.json'))
config['device'] = 'cpu'
config['context'] = 32
config['dim'] = 64
config['n_heads'] = 4
config['depth'] = 2
model = LLMLightning(True, config)
batch = {
    'input_ids': torch.randint(0, config['vocab_size'], (2, config['context'])),
    'attention_mask': torch.ones(2, config['context'])
}
print('forward start')
loss = model.training_step(batch, 0)
print('loss', loss.item())
loss.backward()
print('done backward')
