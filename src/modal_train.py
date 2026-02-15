import modal
import os

app = modal.App("tiny-recursive-lm")

src_path = os.path.dirname(os.path.abspath(__file__))

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "lightning",
        "transformers",
        "datasets",
        "einops",
        "tensorboard",
        "matplotlib",
        "accelerate",
        "wandb",
        "zstandard",
    )
    .add_local_dir(
        src_path,
        remote_path="/root/src",
        ignore=["__pycache__", "*.pyc", ".env", "trm_logs", "*.pt"],
    )
)

vol = modal.Volume.from_name("training-vol", create_if_missing=True)


@app.function(
    image=image,
    gpu="H200",
    volumes={"/results": vol},
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("huggingface-secret")],
    timeout=14400,
)
def train_remote(model: str = "trm", batch_size: int = 4, accumulate_grad_batches: int = 1):
    import sys
    sys.path.insert(0, "/root/src")

    import torch

    torch.set_float32_matmul_precision("high")

    from train import train_llm

    print(f"Model: {model}, Batch size: {batch_size}, Grad accum: {accumulate_grad_batches}")
    train_llm(model, batch_size=batch_size, accumulate_grad_batches=accumulate_grad_batches)

    # Copy checkpoints to volume
    import glob, shutil
    for pt_file in glob.glob("/root/src/trm-*.pt"):
        dest = f"/results/{os.path.basename(pt_file)}"
        shutil.copy2(pt_file, dest)
        print(f"Saved {dest}")
    vol.commit()


@app.local_entrypoint()
def main(model: str = "trm", batch_size: int = 4, accumulate_grad_batches: int = 1):
    train_remote.remote(
        model=model,
        batch_size=batch_size,
        accumulate_grad_batches=accumulate_grad_batches,
    )
