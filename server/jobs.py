"""Job queue: exactly one inference worker, a bounded queue, and abandoned-job cancellation."""
from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

log = logging.getLogger("kshana.jobs")

QUEUE_CAPACITY = 6
JOB_TIMEOUT_S = 30.0
POLL_TIMEOUT_S = 10.0
KEEP_FINISHED_S = 120.0
TERMINAL = {"done", "refused", "error", "cancelled", "timeout"}


class Busy(Exception):
    pass


@dataclass
class Job:
    id: str
    clean: object
    budget: str
    device: object
    created: float
    last_poll: float
    status: str = "queued"
    best: dict | None = None
    version: int = 0
    checklist: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    tier: str | None = None
    reason: str | None = None
    notes: list = field(default_factory=list)
    error: str | None = None
    code: str | None = None
    finished: float | None = None


class JobManager:
    def __init__(
        self,
        pipeline,
        capacity: int = QUEUE_CAPACITY,
        job_timeout_s: float = JOB_TIMEOUT_S,
        poll_timeout_s: float = POLL_TIMEOUT_S,
        clock=time.monotonic,
    ):
        self.pipeline = pipeline
        self.capacity = capacity
        self.job_timeout_s = job_timeout_s
        self.poll_timeout_s = poll_timeout_s
        self.clock = clock
        self.jobs: dict[str, Job] = {}
        self.queue: deque[Job] = deque()
        self.running: Job | None = None
        self.lock = threading.Lock()
        self.wake = threading.Condition(self.lock)
        self.ready = threading.Event()
        self.load_error: str | None = None
        self.boot_s: float | None = None
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None

    # --- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        self._worker = threading.Thread(target=self._run, name="kshana-worker", daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._stop.set()
        with self.wake:
            self.wake.notify_all()
        if self._worker:
            self._worker.join(timeout=5)

    def _boot(self) -> None:
        t0 = time.monotonic()
        try:
            self.pipeline.load()
            costs = self.pipeline.benchmark()
            self.boot_s = round(time.monotonic() - t0, 1)
            log.info("ready in %.1fs, costs %s", self.boot_s, costs.as_dict() if hasattr(costs, "as_dict") else costs)
            self.ready.set()
        except Exception as e:  # health reports it; the page shows the backend-down state
            log.exception("boot failed")
            self.load_error = str(e)

    def _run(self) -> None:
        # Models load on the worker thread, so inference is only ever on this one thread.
        self._boot()
        if not self.ready.is_set():
            return
        while not self._stop.is_set():
            job = self._next()
            if job is not None:
                self._process(job)

    # --- submit and poll ----------------------------------------------------

    def _finish_locked(self, job: Job, status: str, error: str | None = None, code: str | None = None) -> None:
        job.status = status
        job.error = error
        job.code = code
        job.finished = self.clock()

    def _sweep_locked(self) -> None:
        now = self.clock()
        for job in list(self.queue):
            if now - job.last_poll > self.poll_timeout_s:
                self.queue.remove(job)
                self._finish_locked(job, "cancelled", "Cancelled: the page stopped checking on this job.", "abandoned")
            elif now - job.created > self.job_timeout_s:
                self.queue.remove(job)
                self._finish_locked(job, "timeout", "The queue took longer than 30 s. Please try again.", "timeout")
        for job_id, job in list(self.jobs.items()):
            if job.finished is not None and now - job.finished > KEEP_FINISHED_S:
                del self.jobs[job_id]

    def submit(self, clean, budget: str, device) -> Job:
        now = self.clock()
        job = Job(uuid.uuid4().hex, clean, budget, device, now, now)
        with self.wake:
            self._sweep_locked()
            if len(self.queue) >= self.capacity:
                raise Busy()
            self.jobs[job.id] = job
            self.queue.append(job)
            self.wake.notify_all()
        return job

    def add_refused(self, category: str) -> Job:
        """A prompt refused before queueing still gets a job, so the client has one code path."""
        now = self.clock()
        job = Job(uuid.uuid4().hex, None, "", None, now, now)
        with self.lock:
            self._finish_locked(
                job, "refused", category if category.endswith(".") else f"This prompt can't be generated because it asks for {category}.", "prompt_safety"
            )
            self.jobs[job.id] = job
        return job

    def poll(self, job_id: str) -> tuple[Job | None, int]:
        """Return the job and its queue position: 0 running or finished, 1 next in line."""
        with self.lock:
            self._sweep_locked()
            job = self.jobs.get(job_id)
            if job is None:
                return None, 0
            job.last_poll = self.clock()
            position = 0
            if job.status == "queued":
                position = next((i + 1 for i, j in enumerate(self.queue) if j is job), 0)
            return job, position

    def stats(self) -> dict:
        with self.lock:
            return {
                "queue_depth": len(self.queue),
                "queue_capacity": self.capacity,
                "running": self.running is not None,
            }

    # --- worker -------------------------------------------------------------

    def _next(self) -> Job | None:
        with self.wake:
            self._sweep_locked()
            if not self.queue:
                self.wake.wait(timeout=0.5)
                self._sweep_locked()
            if not self.queue:
                return None
            job = self.queue.popleft()
            job.status = "running"
            self.running = job
            return job

    def _process(self, job: Job) -> None:
        now = self.clock()
        deadline = job.created + self.job_timeout_s
        job.timings["queue_ms"] = round((now - job.created) * 1000.0, 1)

        def on_plan(plan, notes):
            with self.lock:
                job.tier, job.reason = plan.tier, plan.reason
                job.notes = list(notes)
                job.timings["plan"] = {"resolution": plan.resolution, "checks": plan.checks, "budget_s": plan.budget_s}

        def on_best(best: dict):
            with self.lock:
                job.checklist = best.pop("checklist")
                job.best = best
                job.version += 1

        def should_stop() -> bool:
            n = self.clock()
            return n - job.last_poll > self.poll_timeout_s or n > deadline

        try:
            with self.lock:
                waiting = len(self.queue)
            outcome = self.pipeline.run(
                job.clean, job.budget, job.device, deadline - now, waiting, on_plan, on_best, should_stop
            )
            with self.lock:
                job.timings.update(outcome.timings)
                job.timings["total_ms"] = round((self.clock() - job.created) * 1000.0, 1)
                status, err, code = outcome.status, outcome.error, outcome.code
                if status == "refused":
                    job.best = None
                # Out of time with an image in hand is still a result.
                if status == "cancelled" and job.best is not None and self.clock() - job.last_poll <= self.poll_timeout_s:
                    status, err, code = "done", None, None
                elif status == "cancelled" and self.clock() - job.last_poll <= self.poll_timeout_s:
                    status, err, code = "timeout", "No image finished within 30 s. Please try again.", "timeout"
                self._finish_locked(job, status, err, code)
        except Exception:
            log.exception("job failed")
            with self.lock:
                job.timings["total_ms"] = round((self.clock() - job.created) * 1000.0, 1)
                self._finish_locked(job, "error", "Something went wrong while generating. Please try again.", "internal")
        finally:
            with self.lock:
                self.running = None
