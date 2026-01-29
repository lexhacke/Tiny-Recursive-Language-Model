"""
TODO:
- Training loop + Linear weight L2 constraint implementation for nGPT
"""

from lightning.pytorch import LightningModule, Trainer
import subprocess
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
from losses import LossModule
from torch import nn
from trm import TinyRecursiveLM
from transformer_baseline import TransformerBaseline
from einops import rearrange
import torch, os, json
from dataset import create_dataset
from pyngrok import ngrok
import torch.nn.functional as F

import torch
torch.set_float32_matmul_precision("high")

class LLMLightning(LightningModule):
    def __init__(self, recursive, config):
        super().__init__()
        self.config = config
        self.slm = TinyRecursiveLM(config) if recursive else TransformerBaseline(config)
        self.lr = config['lr']
        self.nGPT = (config['norm_type'] == 'nGPT')

        self.save_hyperparameters()
        self.loss = LossModule(config)
    
    def on_before_zero_grad(self, optimizer):
        if self.nGPT:
            with torch.no_grad():
                for linear in self.slm.linears:
                    linear.data.copy_(F.normalize(linear, p=2, dim=-1))

    def forward(self, batch):
        return self.slm(batch['input_ids'], batch['attention_mask'])

    def training_step(self, batch, batch_idx):
        pred, conf, _ = self(batch)
        pred, conf = torch.stack(pred), torch.stack(conf)
        bce_loss, ce_loss = self.loss(pred, conf, batch['input_ids'])
        self.log("Confidence_BCE", bce_loss.detach(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("Token_CCE", ce_loss.detach(), on_step=True, on_epoch=True, prog_bar=True)
        return bce_loss + ce_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.95), # Standard LLM betas
            weight_decay=0.1
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
        return [optimizer], [scheduler]

def train_llm(model, ngrok_key, config_name='config.json', launch_subprocesses=True):
    assert model in {'trm', 'baseline'}, "Model must be 'trm' or 'baseline'"
    log_dir = "trm_logs"
    logger = TensorBoardLogger(log_dir, name="slm")

    url = None
    trainer = None
    success = False

    if launch_subprocesses:
        tb_process = subprocess.Popen(['tensorboard', '--logdir', log_dir, '--port', '6006'])
        ngrok.set_auth_token(ngrok_key)
        url = ngrok.connect(6006)

    try:
        print("Tensorboard URL:", url)
        config = json.load(open("config/"+config_name, "r"))
        cuda_config = {x:config[x] for x in config}
        cuda_config['device'] = 'cuda'

        print("Creating model...")
        trm_lightning = LLMLightning(recursive=(model == 'trm'), config=cuda_config)
        print("Creating dataset...")
        ds = create_dataset(cuda_config['context'], cuda_config['tokenizer'])
        print("Creating dataloader...")
        dl = DataLoader(
            ds,
            batch_size=8,
            pin_memory=True,
            persistent_workers=False,
            num_workers=0
        )
        print("Creating trainer...")
        trainer = Trainer(
            max_epochs=1,
            accelerator="gpu",
            devices=1,
            log_every_n_steps=5,
            logger=logger
        )

        print("Starting training...")
        trainer.fit(trm_lightning, dl)
        success = True
        print("Training completed successfully!")

    except Exception as e:
        print(f"ERROR DURING TRAINING: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if url:
            ngrok.disconnect(url)
        if launch_subprocesses:
            tb_process.terminate()
        if trainer:
            torch.save(trainer.model.slm.state_dict(), 'trm-42M-384d.pt')
        print("Terminated Tensorboard Process")

        if success:
            # Only use os._exit if training succeeded (to avoid GIL bug)
            import gc
            del trm_lightning, ds, dl, trainer
            gc.collect()
            torch.cuda.empty_cache()
            import os
            os._exit(0)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    os.environ['NGROK_AUTH']
    train_llm("trm", os.environ['NGROK_AUTH'], launch_subprocesses=False)