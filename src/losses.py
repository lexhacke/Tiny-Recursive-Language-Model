import torch, os ,json
from torch import nn
from einops import rearrange

class LossModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pad_idx = config['pad_idx']
        self.BCE = nn.BCEWithLogitsLoss(reduction='mean')
        self.CE = nn.CrossEntropyLoss(ignore_index=self.pad_idx, reduction='mean')

    def forward(self, token_logits: torch.Tensor, conf_logits: torch.Tensor, token_gt: torch.Tensor):
        """
        Expects:
        token_logits: (T, B, N, D) - predicted token logits
        conf_logits: (T, B, N, 1) - predicted confidence logits
        token_gt: (B, N) - ground truth token ids

        Returns:
        tuple of (BCE confidence loss, CE token prediction loss)
        """
        # Shift for next-token prediction
        token_gt = token_gt[:, 1:]
        token_logits = token_logits[:, :, :-1, :]
        conf_logits = conf_logits[:, :, :-1, :]

        token_ids = token_logits.argmax(dim=-1) # Fake token targets
        conf_gt = (token_ids == token_gt).float() # Fake confidence targets

        conf_gt = rearrange(conf_gt, "... N -> (...) N")
        conf_logits = rearrange(conf_logits.squeeze(-1), "... N -> (...) N")
        token_gt = token_gt.expand(token_logits.shape[0], -1, -1)
        token_gt = rearrange(token_gt, "... -> (...)")
        token_logits = rearrange(token_logits, "... D -> (...) D")
        return self.BCE(conf_logits, conf_gt), self.CE(token_logits, token_gt)

if __name__ == "__main__":
    from trm import TinyRecursiveLM
    from transformers import AutoTokenizer

    config = json.load(open("config/config.json", "r"))

    slm = TinyRecursiveLM(config).to(config['device'])
    tok = AutoTokenizer.from_pretrained(config['tokenizer'])
    
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'right'
    tok.truncation_side = 'right'

    loss_module = LossModule(config)

    batch = tok(["This is an example forward pass. I hope it works man.",
                       "Example 2. Short af."],
                      padding='max_length',
                      truncation=True,
                      max_length=config['context'],
                      return_tensors='pt'
                  )
    
    pred, conf, _ = slm(batch['input_ids'].to(config['device']), batch['attention_mask'].to(config['device']))
    pred, conf = torch.stack(pred), torch.stack(conf)
    bce_loss, ce_loss = loss_module(pred, conf, batch['input_ids'].to(config['device']))
    print(f"BCE Loss: {bce_loss.item()}, CE Loss: {ce_loss.item()}")