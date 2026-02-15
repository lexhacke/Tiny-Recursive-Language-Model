"""
TODO:
- Training loop + Linear weight L2 constraint implementation for nGPT
"""

from lightning.pytorch import LightningModule, Trainer
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from losses import LossModule
from torch import nn
from trm import TinyRecursiveLM
from transformer_baseline import TransformerBaseline
from einops import rearrange
import torch, json, os
from dataset import create_dataset
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
        self.use_muon = (config['optimizer'] == 'muon')
        self.current_max_steps = config.get('max_steps', 1000)
        self.lr_schedule_cfg = config.get('wsd', {}).copy()
        self.total_training_steps = 1
        self.stage_step_offset = 0
        self.accumulate_grad_batches = 1  # updated before each stage

        if self.use_muon:
            self.automatic_optimization = False

        self.save_hyperparameters()
        self.loss = LossModule(config)

    def _normalize_ngpt_weights(self):
        if self.nGPT:
            with torch.no_grad():
                for linear in self.slm.linears:
                    linear.data.copy_(F.normalize(linear, p=2, dim=-1))

    def on_before_zero_grad(self, optimizer):
        if not self.use_muon:
            self._normalize_ngpt_weights()

    def forward(self, batch):
        return self.slm(batch['input_ids'], batch['attention_mask'])

    def training_step(self, batch, batch_idx):
        pred, conf, _ = self(batch)
        pred, conf = torch.stack(pred), torch.stack(conf)
        bce_loss, ce_loss = self.loss(pred, conf, batch['input_ids'])
        loss = bce_loss + ce_loss

        if self.use_muon:
            optimizers = self.optimizers()
            schedulers = self.lr_schedulers()
            if not isinstance(optimizers, (list, tuple)):
                optimizers = [optimizers]
            if not isinstance(schedulers, (list, tuple)):
                schedulers = [schedulers]

            self.manual_backward(loss)
            accum = self.accumulate_grad_batches
            if (batch_idx + 1) % accum == 0:
                for opt in optimizers:
                    opt.step()
                    opt.zero_grad()
                for sch in schedulers:
                    sch.step()
                self._normalize_ngpt_weights()

        self.log("Confidence_BCE", bce_loss.detach(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("Token_CCE", ce_loss.detach(), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        wd = 0.0 if self.nGPT else 0.1

        if self.use_muon:
            muon_params, adam_decay, adam_no_decay = self._split_parameter_groups()
            optimizers = []
            schedulers = []

            if muon_params:
                opt_muon = torch.optim.Muon(
                    muon_params,
                    lr=0.02,
                    momentum=0.95,
                    nesterov=True,
                    ns_steps=5,
                    weight_decay=wd,
                )
                sch_muon = _build_wsd_scheduler(opt_muon, self.lr_schedule_cfg)
                optimizers.append(opt_muon)
                schedulers.append(sch_muon)

            adam_groups = []
            if adam_decay:
                adam_groups.append({"params": adam_decay, "weight_decay": wd})
            if adam_no_decay:
                adam_groups.append({"params": adam_no_decay, "weight_decay": 0.0})
            if adam_groups:
                opt_adam = torch.optim.AdamW(
                    adam_groups,
                    lr=self.lr,
                    betas=(0.9, 0.95),
                )
                sch_adam = _build_wsd_scheduler(opt_adam, self.lr_schedule_cfg)
                optimizers.append(opt_adam)
                schedulers.append(sch_adam)

            return optimizers, schedulers

        decay_params, no_decay_params = self._adam_parameter_groups()
        param_groups = []
        if decay_params:
            param_groups.append({"params": decay_params, "weight_decay": wd})
        if no_decay_params:
            param_groups.append({"params": no_decay_params, "weight_decay": 0.0})

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.lr,
            betas=(0.9, 0.95),
        )
        scheduler = _build_wsd_scheduler(optimizer, self.lr_schedule_cfg)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _split_parameter_groups(self):
        muon_params = []
        adam_decay = []
        adam_no_decay = []
        for name, p in self.slm.named_parameters():
            if not p.requires_grad:
                continue
            is_embedding = 'embedding' in name
            if p.ndim >= 2 and not is_embedding:
                muon_params.append(p)
            elif p.ndim >= 2:
                adam_decay.append(p)
            else:
                adam_no_decay.append(p)
        for p in self.loss.parameters():
            if not p.requires_grad:
                continue
            if p.ndim >= 2:
                adam_decay.append(p)
            else:
                adam_no_decay.append(p)
        return muon_params, adam_decay, adam_no_decay

    def _adam_parameter_groups(self):
        decay_params = []
        no_decay_params = []
        for name, p in self.slm.named_parameters():
            if not p.requires_grad:
                continue
            if 'embedding' in name or p.ndim == 1:
                no_decay_params.append(p)
            else:
                decay_params.append(p)
        for p in self.loss.parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1:
                no_decay_params.append(p)
            else:
                decay_params.append(p)
        return decay_params, no_decay_params

def train_llm(model, config_path=None, batch_size=None, accumulate_grad_batches=1):
    assert model in {'trm', 'baseline'}, "Model must be 'trm' or 'baseline'"

    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "config.json")
    config = json.load(open(config_path, "r"))
    config['device'] = 'cuda'
    if config.get('context') != 4096:
        raise ValueError("SmolLM3 regimen requires context length of 4096 tokens.")
    batch_size = batch_size or config.get('batch_size', 4)

    print("Creating model...")
    trm_lightning = LLMLightning(recursive=(model == 'trm'), config=config)

    if 'curriculum' in config:
        print("[train] Curriculum config detected, but the current training pipeline always streams SlimPajama.")

    _train_single(trm_lightning, config, model, batch_size, accumulate_grad_batches)

def _train_single(trm_lightning, config, model, batch_size, accumulate_grad_batches):
    use_muon = (config['optimizer'] == 'muon')
    trm_lightning.accumulate_grad_batches = accumulate_grad_batches
    single_schedule = config.get('wsd', {}).copy()
    single_schedule.setdefault('total_steps', trm_lightning.current_max_steps)
    single_schedule.setdefault('offset', 0)
    single_schedule.setdefault('final_decay_ratio', 0.10)
    single_schedule.setdefault('min_lr_scale', 0.0)
    trm_lightning.lr_schedule_cfg = single_schedule
    logger = WandbLogger(project="tiny-recursive-lm", name=model)
    ds = create_dataset(config['context'], config['tokenizer'])
    dl = DataLoader(ds, batch_size=batch_size, pin_memory=True, num_workers=0)

    trainer_kwargs = dict(
        max_epochs=1,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        log_every_n_steps=5,
        logger=logger,
    )
    if not use_muon:
        trainer_kwargs['accumulate_grad_batches'] = accumulate_grad_batches

    trainer = Trainer(**trainer_kwargs)
    trainer.fit(trm_lightning, dl)
    torch.save(trainer.model.slm.state_dict(), 'trm-checkpoint.pt')
    print("Training finished.")

def _build_wsd_scheduler(optimizer, cfg):
    total_steps = max(1, int(cfg.get('total_steps', 1)))
    warmup_steps = cfg.get('warmup_steps')
    if warmup_steps is None:
        warmup_ratio = cfg.get('warmup_ratio', 0.04)
        warmup_steps = int(total_steps * warmup_ratio)
    warmup_steps = max(1, int(warmup_steps))
    final_decay_ratio = cfg.get('final_decay_ratio', 0.10)
    min_lr = cfg.get('min_lr_scale', 0.0)
    offset = int(cfg.get('offset', 0))

    decay_start = max(warmup_steps, int(total_steps * (1 - final_decay_ratio)))

    def lr_lambda(local_step):
        global_step = offset + local_step
        if global_step < warmup_steps:
            return global_step / max(1, warmup_steps)
        if global_step < decay_start:
            return 1.0
        if global_step >= total_steps:
            return min_lr
        progress = (global_step - decay_start) / max(1, total_steps - decay_start)
        return 1.0 - (1.0 - min_lr) * min(progress, 1.0)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

if __name__ == "__main__":
    train_llm("trm")
