"""FastAPI app. Local: uvicorn server.app:create_app --factory --port 8000"""
from __future__ import annotations

import logging
import os
import resource
import sys
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from kshana.inputs import InputError, sanitize
from kshana.planner import Device
from kshana.safety import check_prompt

from .jobs import Busy, JobManager

logging.basicConfig(level=logging.INFO)

RATE_LIMIT = 5
RATE_WINDOW_S = 60.0
WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class DeviceIn(BaseModel):
    memory_gb: float | None = Field(default=None, ge=0, le=4096)
    cores: int | None = Field(default=None, ge=0, le=4096)
    save_data: bool = False
    profile: Literal["midrange", "lowend", "oldphone"] | None = None


class GenerateIn(BaseModel):
    prompt: str = Field(max_length=4000)
    budget: Literal["fast", "balanced", "best", "auto"] = "auto"
    device: DeviceIn = DeviceIn()


class RateLimiter:
    def __init__(self, limit: int = RATE_LIMIT, window_s: float = RATE_WINDOW_S, clock=time.monotonic):
        self.limit, self.window_s, self.clock = limit, window_s, clock
        self.hits: dict[str, deque] = defaultdict(deque)

    def check(self, key: str) -> float | None:
        """Record a hit. Returns seconds to wait if over the limit, otherwise None."""
        now = self.clock()
        q = self.hits[key]
        while q and now - q[0] > self.window_s:
            q.popleft()
        if len(q) >= self.limit:
            return self.window_s - (now - q[0])
        q.append(now)
        if len(self.hits) > 10000:  # bound memory under address churn
            for k in [k for k, v in self.hits.items() if not v or now - v[-1] > self.window_s]:
                del self.hits[k]
        return None


def peak_rss_mb() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024))


def client_ip(request: Request) -> str:
    # Cloudflare overwrites this header, but only trust it when a tunnel is the sole way in.
    if os.environ.get("KSHANA_BEHIND_CLOUDFLARE") == "1":
        cf = request.headers.get("cf-connecting-ip")
        if cf:
            return cf.strip()
    # Behind Modal's proxy the last X-Forwarded-For entry is the one the proxy added.
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


def err(status: int, code: str, message: str, retry_after: float | None = None) -> JSONResponse:
    body: dict = {"error": {"code": code, "message": message}}
    headers = {}
    if retry_after is not None:
        body["retry_after"] = max(1, round(retry_after))
        headers["Retry-After"] = str(body["retry_after"])
    return JSONResponse(body, status_code=status, headers=headers)


def job_body(job, position: int, since: int | None) -> dict:
    best = None
    if job.best is not None:
        best = {k: v for k, v in job.best.items() if k in {"mime", "width", "height", "seed", "attempt", "score"}}
        best["version"] = job.version
        # Polling once a second must not resend an unchanged image over mobile data.
        if since != job.version:
            best["image_b64"] = job.best["image_b64"]
    timings = dict(job.timings)
    if job.best is not None:
        timings["last_attempt"] = job.best.get("attempt_times")
    return {
        "job_id": job.id,
        "status": job.status,
        "queue_position": position,
        "best": best,
        "checklist": job.checklist,
        "classical": job.best.get("classical") if job.best else [],
        "quality": job.best.get("quality") if job.best else None,
        "timings": timings,
        "tier": job.tier,
        "reason": job.reason,
        "notes": job.notes,
        "error": job.error,
        "code": job.code,
    }


def create_app(pipeline=None, manager: JobManager | None = None, limiter: RateLimiter | None = None) -> FastAPI:
    if manager is None:
        if pipeline is None:
            from kshana.pipeline import Pipeline

            pipeline = Pipeline()
        manager = JobManager(pipeline)
    limiter = limiter or RateLimiter()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        manager.start()
        yield
        manager.stop()

    app = FastAPI(title="Kshana", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.manager = manager

    origin = os.environ.get("KSHANA_CORS_ORIGIN")
    if origin:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[origin],
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
            max_age=600,
        )

    @app.exception_handler(RequestValidationError)
    async def bad_request(_: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", [])[1:]) or "request"
        return err(422, "bad_request", f"Invalid {where}: {first.get('msg', 'bad value')}.")

    @app.get("/health")
    def health():
        stats = manager.stats()
        if manager.load_error:
            return JSONResponse({"status": "error", "ready": False, **stats}, status_code=503)
        ready = manager.ready.is_set()
        body = {"status": "ok" if ready else "warming", "ready": ready, **stats, "boot_s": manager.boot_s, "peak_rss_mb": peak_rss_mb()}
        gen = getattr(manager.pipeline, "gen", None)
        if gen is not None:
            body["unet_backend"] = gen.backend
        costs = getattr(manager.pipeline, "costs", None)
        if ready and costs is not None and hasattr(costs, "as_dict"):
            body["costs_ms"] = costs.as_dict()
        return body

    @app.post("/api/generate")
    def generate(body: GenerateIn, request: Request):
        if not manager.ready.is_set():
            if manager.load_error:
                return err(503, "backend_error", "The server failed to start.")
            return err(503, "warming", "The server is warming up. It will be ready shortly.", 5)
        try:
            clean = sanitize(body.prompt)
        except InputError as e:
            return err(422, e.code, e.message)
        wait = limiter.check(client_ip(request))
        if wait is not None:
            return err(429, "rate_limited", f"That's {RATE_LIMIT} images this minute, the limit per person.", wait)
        category = check_prompt(clean.text)
        if category:
            return {"job_id": manager.add_refused(category).id}
        device = Device(body.device.memory_gb, body.device.cores, body.device.save_data, body.device.profile)
        try:
            job = manager.submit(clean, body.budget, device)
        except Busy:
            return err(503, "busy", "The server is busy with other people's images. Please try again soon.", 15)
        return {"job_id": job.id}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str, since: int | None = None):
        job, position = manager.poll(job_id)
        if job is None:
            return err(404, "unknown_job", "This job is gone. The server may have restarted. Please try again.")
        return job_body(job, position, since)

    # Same-origin page for `docker run` and local use. GitHub Pages serves the same folder for the public URL.
    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
