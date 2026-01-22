from lightning.pytorch import LightningModule, Trainer
import subprocess
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn
from trm import TinyRecursiveLM
from einops import rearrange
import torch, os, json
from dataset import create_dataset
from pyngrok import ngrok

class LLMLightning(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.slm = TinyRecursiveLM(config) #TransformerBaseline(config)
        self.lr = config['lr']
        self.CE = nn.CrossEntropyLoss(ignore_index=config['pad_idx'], reduction='mean')

    def forward(self, batch):
        out = self.slm(batch['input_ids'], batch['attention_mask'])
        if isinstance(out, tuple):
            out = out[0]
        return out

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_labels = batch['input_ids'][..., 1:].contiguous()

        shifted_logits = rearrange(shifted_logits, "B N D -> (B N) D")
        shifted_labels = rearrange(shifted_labels, "B N -> (B N)")

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("Failure on batch", batch_idx)
            raise RuntimeError("logits contain NaN/Inf")

        loss = self.CE(shifted_logits, shifted_labels)

        self.log("CCE", loss.detach(), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.95), # Standard LLM betas
            weight_decay=0.1
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
        return [optimizer], [scheduler]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def train_llm(ngrok_key, config_name='config.json', colab=True):
    logger = TensorBoardLogger("trm_logs", name="slm")
    tb_process = subprocess.Popen(['tensorboard', '--logdir', 'trm_logs', '--port', '6006'])
    url = None
    trainer = None
    try:
        ngrok.set_auth_token(ngrok_key)
        url = ngrok.connect(6006)
        print("Tensorboard URL:", url)
        config_path = os.path.join(SCRIPT_DIR, "config", config_name)
        config = json.load(open(config_path, "r"))
        cuda_config = {x:config[x] for x in config}
        cuda_config['device'] = 'cuda'

        trm_lightning = LLMLightning(cuda_config).to('cuda')
        ds = create_dataset(cuda_config['context'], cuda_config['tokenizer'])
        dl = DataLoader(
            ds,
            batch_size=16,
            pin_memory=True,
            persistent_workers=True,
            num_workers=8
        )
        trainer = Trainer(
            max_epochs=1,
            accelerator="gpu",
            devices=1,
            log_every_n_steps=5,
            logger=logger
        )

        trainer.fit(trm_lightning, dl)
    finally:
        if url:
            ngrok.disconnect(url)
        tb_process.terminate()
        if trainer:
            trainer.model.slm.save_state_dict('trm-42M-384d.pt')
        print("Terminated Tensorboard Process")