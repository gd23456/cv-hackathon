"""Modal deployment. Deploy from the repo root:

    KSHANA_CORS_ORIGIN=https://gd23456.github.io modal deploy server/modal_app.py

Redeploying resets autoscaler overrides and snapshots: freeze after submission.
"""
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
APP_NAME = "kshana"

app = modal.App(APP_NAME)
image = modal.Image.from_dockerfile(ROOT / "Dockerfile", context_dir=ROOT)
cron_image = modal.Image.debian_slim()


@app.function(
    image=image,
    cpu=(2.0, 4.0),  # physical cores: 4 vCPU requested, up to 8. Warm cost follows the request.
    memory=6144,
    min_containers=1,
    max_containers=2,
    timeout=60,
    startup_timeout=300,
    env={
        "KSHANA_CORS_ORIGIN": os.environ.get("KSHANA_CORS_ORIGIN", "https://gd23456.github.io"),
        "KSHANA_THREADS": "4",
    },
)
# Jobs live in this process's memory, so polls must reach the container that took the job.
# A high input limit keeps all traffic on one container; the second is failover only.
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web():
    from server.app import create_app

    return create_app()


# Tue 10 Nov 2026 06:00 IST: stop paying for the warm container.
@app.function(image=cron_image, schedule=modal.Cron("0 6 10 11 *", timezone="Asia/Kolkata"), cpu=0.25, memory=256)
def warm_off():
    modal.Function.from_name(APP_NAME, "web").update_autoscaler(min_containers=0)
