"""Baseline image generation on Modal: detached, capped, budget-guarded, resumable. Blocked so far: the first launch was refused because the workspace is at its spend limit.

    modal run --detach eval/modal_batch.py --config B0-T4 --split coco_5k
    python eval/run.py --config B0-T4 --split coco_5k --inception --from-volume kshana-eval

The driver runs remotely, so closing the laptop does not stop it. Finished PNGs on the volume are skipped on retry.
Spend is recorded in a modal.Dict shared by all runs; dispatch stops before the total could pass BUDGET_USD,
counting every running worker at its worst case.
"""
import json
import sys
import time
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from budget import BudgetGuard, Rates, worker_rate  # noqa: E402

BUDGET_USD = 12.0
MAX_CONTAINERS = 4
CHUNK = 64
BATCH = {"B0-T4": 8, "B1-T4": 16}
CHUNK_TIMEOUT_S = 1800
STARTUP_CHARGE_S = 420  # image pull, model load, idle window: charged per container, on the high side
WORKER_CPUS = 2.0
WORKER_MEM_GIB = 12.0
RATE = worker_rate(Rates(), WORKER_CPUS, WORKER_MEM_GIB)
DRIVER_TIMEOUT_S = min(24 * 3600, int(BUDGET_USD / (RATE * MAX_CONTAINERS)))  # wall-clock backstop
JOBS = {
    "B0-T4": {"repo": "stable-diffusion-v1-5/stable-diffusion-v1-5", "steps": 25, "cfg": 7.5},
    "B1-T4": {"repo": "stabilityai/sd-turbo", "steps": 1, "cfg": 0.0},
}
ALLOW = [
    "unet/*.fp16.safetensors", "unet/config.json", "text_encoder/*.fp16.safetensors", "text_encoder/config.json",
    "vae/*.fp16.safetensors", "vae/config.json", "tokenizer/*", "scheduler/*", "model_index.json",
]

app = modal.App("kshana-eval-batch")
volume = modal.Volume.from_name("kshana-eval", create_if_missing=True)
ledger = modal.Dict.from_name("kshana-eval-ledger", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.7.0", "numpy==2.2.6", "diffusers==0.42.1", "transformers==5.22.1", "accelerate==1.15.0", "safetensors==0.8.0", "huggingface_hub==2.4.0", "pillow")
    .add_local_file(str(Path(__file__).resolve().parent / "budget.py"), "/root/budget.py")
)


@app.cls(gpu="T4", cpu=WORKER_CPUS, memory=int(WORKER_MEM_GIB * 1024), image=image, volumes={"/vol": volume},
         max_containers=MAX_CONTAINERS, scaledown_window=60, timeout=CHUNK_TIMEOUT_S)
class Worker:
    config: str = modal.parameter()

    @modal.enter()
    def load(self):
        import torch
        from diffusers import StableDiffusionPipeline
        from huggingface_hub import snapshot_download

        job = JOBS[self.config]
        path = snapshot_download(job["repo"], local_dir=f"/vol/hf/{job['repo']}", token=False, allow_patterns=ALLOW)
        self.pipe = StableDiffusionPipeline.from_pretrained(
            path, variant="fp16", use_safetensors=True, torch_dtype=torch.float32, safety_checker=None, feature_extractor=None,
            requires_safety_checker=False,
        ).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.enable_vae_slicing()  # one image at a time through the decoder; output is unchanged
        self.job = job

    @modal.method()
    def run_chunk(self, split: str, rows: list[dict]) -> dict:
        import numpy as np
        import torch
        from PIL import Image

        t0 = time.time()
        out_dir = Path(f"/vol/gen/{self.config}/{split}")
        out_dir.mkdir(parents=True, exist_ok=True)
        volume.reload()
        rows = [r for r in rows if not (out_dir / f"{r['id']}.png").exists()]
        size = BATCH[self.config]
        for start in range(0, len(rows), size):
            batch = rows[start : start + size]
            z = torch.cat([torch.randn((1, 4, 64, 64), generator=torch.Generator("cpu").manual_seed(int(r["seed"])), dtype=torch.float32) for r in batch])
            images = self.pipe([r["prompt"] for r in batch], num_inference_steps=self.job["steps"], guidance_scale=self.job["cfg"],
                               latents=z.to("cuda"), output_type="np").images
            for r, arr in zip(batch, images):
                Image.fromarray((arr * 255).round().astype(np.uint8)).save(out_dir / f"{r['id']}.png")
        volume.commit()
        return {"seconds": time.time() - t0, "images": len(rows)}


@app.function(image=image, volumes={"/vol": volume}, timeout=DRIVER_TIMEOUT_S, cpu=0.25, memory=512)
def driver(config: str, split: str, rows: list[dict]) -> dict:
    volume.reload()
    out_dir = Path(f"/vol/gen/{config}/{split}")
    finished = {p.stem for p in out_dir.glob("*.png")} if out_dir.exists() else set()
    rows = [r for r in rows if r["id"] not in finished]
    chunks = [rows[i : i + CHUNK] for i in range(0, len(rows), CHUNK)]
    print(f"{config}/{split}: {len(finished)} finished, {len(rows)} to do, {len(chunks)} chunks, spent ${ledger.get('spent_usd', 0.0):.2f}", flush=True)

    guard = BudgetGuard(BUDGET_USD, ledger.get("spent_usd", 0.0), CHUNK_TIMEOUT_S * RATE, STARTUP_CHARGE_S * RATE)
    worker = Worker(config=config)
    running: list = []
    started = 0
    next_chunk = 0
    done_images = 0
    stopped = False

    def account(usd: float) -> None:
        guard.charge(usd)
        ledger["spent_usd"] = guard.spent_usd

    while next_chunk < len(chunks) or running:
        while next_chunk < len(chunks) and len(running) < MAX_CONTAINERS and not stopped:
            new_container = started < MAX_CONTAINERS
            if not guard.may_dispatch(len(running), new_container):
                print(f"budget guard: stopping dispatch at ${guard.spent_usd:.2f} of ${BUDGET_USD:.2f}", flush=True)
                stopped = True
                break
            if new_container:
                account(guard.startup_usd)
                started += 1
            running.append(worker.run_chunk.spawn(split, chunks[next_chunk]))
            next_chunk += 1
        if stopped and not running:
            break
        still = []
        for call in running:
            try:
                res = call.get(timeout=5)
            except TimeoutError:
                still.append(call)
                continue
            except Exception as exc:  # a failed chunk is charged at worst case and left for the retry to pick up
                print(f"chunk failed: {exc}", flush=True)
                account(guard.worst_chunk_usd)
                continue
            account(res["seconds"] * RATE)
            done_images += res["images"]
            print(f"{done_images} images, ${guard.spent_usd:.2f} spent", flush=True)
        running = still
    return {"images_this_run": done_images, "spent_usd": guard.spent_usd, "stopped_by_budget": stopped}


@app.local_entrypoint()
def main(config: str, split: str, limit: int = 0):
    if config not in JOBS:
        raise SystemExit(f"unknown config {config}; known: {sorted(JOBS)}")
    path = Path(__file__).resolve().parent / "splits" / f"{split}.jsonl"
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if split == "complex_test":
        raise SystemExit("complex_test is held out; generate it only for the final report, by editing this guard deliberately")
    print(driver.remote(config, split, rows[: limit or None]))
