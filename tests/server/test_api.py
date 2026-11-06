"""API and job-queue behaviour against a fake pipeline, so no weights are loaded."""
import threading
import time

import pytest
from fastapi.testclient import TestClient

from kshana.pipeline import Outcome
from kshana.planner import Plan
from server.app import RateLimiter, create_app
from server.jobs import JobManager


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class FakePipeline:
    """Runs until released or told to stop, like the anytime loop."""

    costs = None

    def __init__(self, hold=False, fail=False, load_error=None):
        self.release = threading.Event()
        if not hold:
            self.release.set()
        self.fail = fail
        self.load_error = load_error
        self.threads = set()
        self.started = threading.Event()
        self.runs = []
        self.waiting_seen = []

    def load(self):
        if self.load_error:
            raise RuntimeError(self.load_error)

    def benchmark(self):
        return {}

    def run(self, clean, budget, device, remaining_s, waiting, on_plan, on_best, should_stop):
        self.threads.add(threading.current_thread().name)
        self.started.set()
        self.runs.append(clean.text)
        self.waiting_seen.append(waiting)
        if self.fail:
            raise RuntimeError("boom")
        on_plan(Plan("Verified", 512, "full", 10.0, 3, 1000.0, 512, "Verified: because."), list(clean.notes))
        on_best(
            {
                "image_b64": "QUJD",
                "mime": "image/jpeg",
                "width": 512,
                "height": 512,
                "seed": 1,
                "attempt": 1,
                "score": 0.5,
                "checklist": [{"kind": "presence", "label": "dog", "status": "pass", "detail": ""}],
                "classical": [],
                "quality": {},
                "attempt_times": {},
            }
        )
        while not self.release.wait(0.01):
            if should_stop():
                return Outcome("cancelled", "stopped", "abandoned", {"attempts": 1})
        return Outcome("done", timings={"attempts": 1})


@pytest.fixture
def make_client():
    clients = []

    def build(pipeline=None, limit=1000, capacity=6, poll_timeout_s=10.0, job_timeout_s=30.0, clock=None):
        pipeline = pipeline or FakePipeline()
        manager = JobManager(pipeline, capacity=capacity, poll_timeout_s=poll_timeout_s, job_timeout_s=job_timeout_s, **({"clock": clock} if clock else {}))
        app = create_app(manager=manager, limiter=RateLimiter(limit=limit))
        client = TestClient(app)
        client.__enter__()
        clients.append((client, pipeline))
        if not pipeline.load_error:
            assert manager.ready.wait(5)
        client.manager, client.pipeline = manager, pipeline
        return client

    yield build
    for client, pipeline in clients:
        pipeline.release.set()
        client.__exit__(None, None, None)


def generate(client, prompt="a dog", budget="auto", **kw):
    return client.post("/api/generate", json={"prompt": prompt, "budget": budget, "device": {"memory_gb": 8, "cores": 8, "save_data": False}}, **kw)


def wait_status(client, job_id, want, timeout=5):
    t0 = time.time()
    while time.time() - t0 < timeout:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in want:
            return body
        time.sleep(0.01)
    raise AssertionError(f"never reached {want}: {body}")


def test_health_reports_ready(make_client):
    body = make_client().get("/health").json()
    assert body["ready"] is True and body["status"] == "ok" and body["queue_capacity"] == 6


def test_health_reports_failed_boot(make_client):
    client = make_client(FakePipeline(load_error="no weights"))
    t0 = time.time()
    while time.time() - t0 < 5 and not client.manager.load_error:
        time.sleep(0.01)
    res = client.get("/health")
    assert res.status_code == 503 and res.json()["status"] == "error"
    assert generate(client).status_code == 503


def test_generate_returns_job_with_full_contract(make_client):
    client = make_client()
    res = generate(client)
    assert res.status_code == 200 and set(res.json()) == {"job_id"}
    body = wait_status(client, res.json()["job_id"], {"done"})
    for key in ("status", "queue_position", "best", "checklist", "timings", "tier", "reason", "error"):
        assert key in body
    assert body["best"]["image_b64"] == "QUJD"
    assert body["tier"] == "Verified" and body["reason"]
    assert body["checklist"][0]["label"] == "dog"
    assert "total_ms" in body["timings"] and "queue_ms" in body["timings"]


def test_unchanged_image_is_not_resent(make_client):
    client = make_client()
    job = generate(client).json()["job_id"]
    body = wait_status(client, job, {"done"})
    version = body["best"]["version"]
    again = client.get(f"/api/jobs/{job}", params={"since": version}).json()
    assert "image_b64" not in again["best"]


@pytest.mark.parametrize(
    "prompt,code",
    [("", "empty_prompt"), ("   ", "empty_prompt"), ("一只红色的猫坐在椅子上", "non_english"), ("x " * 900, "too_long")],
)
def test_bad_input_is_a_clear_422(make_client, prompt, code):
    res = generate(make_client(), prompt)
    assert res.status_code == 422 and res.json()["error"]["code"] == code


def test_bad_budget_and_missing_prompt_use_same_error_shape(make_client):
    client = make_client()
    for payload in ({"prompt": "a dog", "budget": "ultra"}, {"budget": "fast"}, {"prompt": "x" * 5000}):
        res = client.post("/api/generate", json=payload)
        assert res.status_code == 422 and res.json()["error"]["code"] == "bad_request"


def test_blocked_prompt_is_refused_without_reaching_the_worker(make_client):
    client = make_client()
    res = generate(client, "a nude person")
    assert res.status_code == 200
    body = client.get(f"/api/jobs/{res.json()['job_id']}").json()
    assert body["status"] == "refused" and body["code"] == "prompt_safety" and body["best"] is None
    assert client.pipeline.runs == []


def test_unknown_job_is_404(make_client):
    res = make_client().get("/api/jobs/nope")
    assert res.status_code == 404 and res.json()["error"]["code"] == "unknown_job"


def test_rate_limit_is_per_ip_and_polling_is_exempt(make_client):
    client = make_client(limit=5)
    jobs = [generate(client, headers={"X-Forwarded-For": "1.1.1.1"}) for _ in range(5)]
    assert all(r.status_code == 200 for r in jobs)
    res = generate(client, headers={"X-Forwarded-For": "1.1.1.1"})
    assert res.status_code == 429 and res.json()["error"]["code"] == "rate_limited" and res.json()["retry_after"] > 0
    assert generate(client, headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 200
    for _ in range(20):
        assert client.get(f"/api/jobs/{jobs[0].json()['job_id']}").status_code == 200


def test_last_forwarded_entry_counts_so_spoofing_does_not_dodge_the_limit(make_client):
    client = make_client(limit=1)
    assert generate(client, headers={"X-Forwarded-For": "9.9.9.1, 3.3.3.3"}).status_code == 200
    assert generate(client, headers={"X-Forwarded-For": "9.9.9.2, 3.3.3.3"}).status_code == 429


def test_cloudflare_address_is_used_only_when_told_to_trust_it(make_client, monkeypatch):
    client = make_client(limit=1)
    a, b = {"CF-Connecting-IP": "5.5.5.5"}, {"CF-Connecting-IP": "6.6.6.6"}
    assert generate(client, headers=a).status_code == 200
    assert generate(client, headers=b).status_code == 429  # same socket address: the header is ignored
    monkeypatch.setenv("KSHANA_BEHIND_CLOUDFLARE", "1")
    assert generate(client, headers=a).status_code == 200
    assert generate(client, headers=a).status_code == 429
    assert generate(client, headers=b).status_code == 200


def test_invalid_prompt_does_not_use_up_the_rate_limit(make_client):
    client = make_client(limit=1)
    assert generate(client, "   ").status_code == 422
    assert generate(client).status_code == 200


def test_queue_is_bounded_with_a_clear_busy_response(make_client):
    client = make_client(FakePipeline(hold=True), capacity=2)
    first = generate(client)
    assert client.pipeline.started.wait(5)
    second, third = generate(client), generate(client)
    assert first.status_code == second.status_code == third.status_code == 200
    res = generate(client)
    assert res.status_code == 503
    assert res.json()["error"]["code"] == "busy" and res.json()["retry_after"] > 0 and res.headers["Retry-After"]
    assert client.get(f"/api/jobs/{second.json()['job_id']}").json()["queue_position"] == 1
    assert client.get(f"/api/jobs/{third.json()['job_id']}").json()["queue_position"] == 2
    client.pipeline.release.set()
    for r in (first, second, third):
        wait_status(client, r.json()["job_id"], {"done"})


def test_exactly_one_worker_thread_runs_jobs(make_client):
    client = make_client()
    jobs = [generate(client, f"a dog {i}").json()["job_id"] for i in range(5)]
    for j in jobs:
        wait_status(client, j, {"done"})
    assert client.pipeline.threads == {"kshana-worker"}
    assert client.pipeline.runs == [f"a dog {i}" for i in range(5)]


def test_planner_is_told_how_many_are_waiting(make_client):
    client = make_client(FakePipeline(hold=True))
    first = generate(client).json()["job_id"]
    assert client.pipeline.started.wait(5)
    others = [generate(client, f"a cat {i}").json()["job_id"] for i in range(3)]
    client.pipeline.release.set()
    for j in [first, *others]:
        wait_status(client, j, {"done"})
    assert client.pipeline.waiting_seen == [0, 2, 1, 0]


def test_running_job_is_cancelled_after_silence(make_client):
    client = make_client(FakePipeline(hold=True), poll_timeout_s=0.1)
    job = generate(client).json()["job_id"]
    assert client.pipeline.started.wait(5)
    time.sleep(0.3)
    body = client.get(f"/api/jobs/{job}").json()
    # The poll above arrives after the cancel, so the state is final.
    t0 = time.time()
    while body["status"] == "running" and time.time() - t0 < 2:
        time.sleep(0.05)
        body = client.get(f"/api/jobs/{job}").json()
    assert body["status"] in {"cancelled", "done"}


def test_queued_job_is_dropped_after_silence_and_frees_its_slot(make_client):
    clock = FakeClock()
    client = make_client(FakePipeline(hold=True), capacity=1, clock=clock)
    first = generate(client).json()["job_id"]
    assert client.pipeline.started.wait(5)
    queued = generate(client).json()["job_id"]
    assert generate(client).status_code == 503
    clock.t += 11
    client.manager.jobs[first].last_poll = clock.t  # the running job is still being polled
    body = client.get(f"/api/jobs/{queued}").json()
    assert body["status"] == "cancelled" and body["code"] == "abandoned"
    assert generate(client).status_code == 200


def test_queued_job_times_out_at_thirty_seconds(make_client):
    clock = FakeClock()
    client = make_client(FakePipeline(hold=True), capacity=1, clock=clock, poll_timeout_s=1000)
    generate(client)
    assert client.pipeline.started.wait(5)
    queued = generate(client).json()["job_id"]
    clock.t += 31
    body = client.get(f"/api/jobs/{queued}").json()
    assert body["status"] == "timeout" and body["code"] == "timeout"


def test_running_job_past_thirty_seconds_keeps_its_image(make_client):
    clock = FakeClock()
    client = make_client(FakePipeline(hold=True), clock=clock)
    job = generate(client).json()["job_id"]
    assert client.pipeline.started.wait(5)
    clock.t += 31
    client.manager.jobs[job].last_poll = clock.t
    body = wait_status(client, job, {"done"})
    assert body["best"]["image_b64"] == "QUJD"


def test_pipeline_crash_becomes_error_and_worker_survives(make_client):
    pipeline = FakePipeline(fail=True)
    client = make_client(pipeline)
    body = wait_status(client, generate(client).json()["job_id"], {"error"})
    assert body["error"] and "boom" not in body["error"]
    pipeline.fail = False
    wait_status(client, generate(client).json()["job_id"], {"done"})


def test_cors_allows_only_the_configured_origin(monkeypatch):
    monkeypatch.setenv("KSHANA_CORS_ORIGIN", "https://team.github.io")
    with TestClient(create_app(manager=JobManager(FakePipeline()))) as client:
        ok = client.options("/api/generate", headers={"Origin": "https://team.github.io", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"})
        assert ok.headers["access-control-allow-origin"] == "https://team.github.io"
        bad = client.options("/api/generate", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
        assert "access-control-allow-origin" not in bad.headers


def test_no_cors_header_without_configuration(monkeypatch):
    monkeypatch.delenv("KSHANA_CORS_ORIGIN", raising=False)
    with TestClient(create_app(manager=JobManager(FakePipeline()))) as client:
        res = client.get("/health", headers={"Origin": "https://team.github.io"})
        assert "access-control-allow-origin" not in res.headers


def test_web_page_is_served(make_client):
    res = make_client().get("/")
    assert res.status_code == 200 and "Kshana" in res.text
