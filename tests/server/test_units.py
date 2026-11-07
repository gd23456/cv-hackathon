"""Parser, inputs, planner, anytime loop, classical layer, verifier rules, safety. No model weights."""
import numpy as np
import pytest

from kshana import classical, generator, parser, planner, safety, verifier, weights
from kshana.anytime import Attempt, run_anytime
from kshana.inputs import CleanPrompt, InputError, fit_tokens, sanitize


# --- backend choice -------------------------------------------------------


def test_unet_backend_defaults_to_low_memory_and_rejects_typos(monkeypatch):
    monkeypatch.delenv("KSHANA_UNET_BACKEND", raising=False)
    assert generator.resolve_backend() == generator.ONNXRUNTIME
    monkeypatch.setenv("KSHANA_UNET_BACKEND", "openvino")
    assert generator.resolve_backend() == generator.OPENVINO
    with pytest.raises(ValueError):
        generator.resolve_backend("tensorflow")


# --- inputs ---------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "   ", "\n\t", "...!!!", None])
def test_empty_prompts_are_rejected(bad):
    with pytest.raises(InputError) as e:
        sanitize(bad)
    assert e.value.code == "empty_prompt"


def test_too_long_is_rejected():
    with pytest.raises(InputError) as e:
        sanitize("cat " * 400)
    assert e.value.code == "too_long"


@pytest.mark.parametrize("text", ["一只红色的猫坐在椅子上", "एक लाल सेब मेज पर", "красная кошка на стуле"])
def test_non_latin_is_rejected(text):
    with pytest.raises(InputError) as e:
        sanitize(text)
    assert e.value.code == "non_english"


def test_latin_non_english_warns_but_generates():
    clean = sanitize("gato rojo sentado silla")
    assert clean.text == "gato rojo sentado silla"
    assert any("English" in n for n in clean.notes)


def test_english_gets_no_warning_and_whitespace_collapses():
    clean = sanitize("  a red   apple\n on a table ")
    assert clean.text == "a red apple on a table"
    assert clean.notes == []


def test_fit_tokens_reports_what_was_dropped():
    clean = fit_tokens(CleanPrompt("keep this drop that"), lambda t: ("keep this", "drop that"))
    assert clean.text == "keep this"
    assert "drop that" in clean.notes[0]


# --- parser ---------------------------------------------------------------


def test_parser_objects_colours_and_relation():
    p = parser.parse("a red apple to the left of a blue cup on a wooden table")
    assert [(o.noun, o.coco, o.colour) for o in p.objects[:2]] == [("apple", "apple", "red"), ("cup", "cup", "blue")]
    assert p.objects[2].coco == "dining table"
    r = p.relations[0]
    assert (p.objects[r.subject].noun, r.relation, p.objects[r.target].noun) == ("apple", "left of", "cup")
    assert p.names_colours


def test_parser_counts_words_and_digits():
    p = parser.parse("three cats and 2 dogs")
    assert {o.coco: o.count for o in p.objects} == {"cat": 3, "dog": 2}


def test_parser_maps_non_coco_to_none():
    p = parser.parse("a small cabin beside a lake")
    assert all(o.coco is None for o in p.objects)
    assert p.relations[0].relation == "next to"


def test_parser_orange_fruit_vs_colour():
    assert parser.parse("an orange on a table").objects[0].coco == "orange"
    cat = parser.parse("an orange cat").objects[0]
    assert (cat.coco, cat.colour) == ("cat", "orange")


def test_golden_is_not_a_colour_check_but_counts_as_naming_colour():
    p = parser.parse("a golden retriever puppy")
    assert p.objects[0].colour is None
    assert p.names_colours


def test_complexity_grows_with_constraints():
    assert parser.parse("a cat").complexity < parser.parse("three red cats to the left of a blue car").complexity


# --- planner --------------------------------------------------------------


COSTS = planner.CostTable(
    text_ms=100, seed_ms={512: 3000, 384: 1500, 256: 700}, classical_ms=20, detect_ms=200, clip_image_ms=50, clip_text_ms=30, threads=4
)
DEVICE = planner.Device(8, 8, False)


def plan(budget="balanced", device=DEVICE, complexity=0.1, waiting=0, remaining=30.0, costs=COSTS):
    return planner.plan_job(costs, budget, device, complexity, waiting, remaining)


def test_simulated_devices_step_down_and_say_why():
    fast = planner.CostTable(100, {512: 1000, 384: 550, 256: 270}, 20, 60, 20, 10)
    tiers = {
        name: plan("best", device=planner.Device(8, 8, False, name), costs=fast)
        for name in ("midrange", "lowend", "oldphone")
    }
    assert tiers["midrange"].tier == planner.STANDARD
    assert tiers["lowend"].tier == planner.CONSTRAINED
    assert tiers["oldphone"].tier == planner.SURVIVAL
    assert "low-end" in tiers["lowend"].reason


def test_tiers_follow_available_time():
    assert plan("best").tier == planner.VERIFIED
    assert plan("fast", costs=planner.CostTable(100, {512: 3000, 384: 1500, 256: 700}, 20, 200, 50, 30)).tier == planner.STANDARD
    assert plan("balanced", waiting=5, remaining=2.4).tier == planner.CONSTRAINED
    assert plan("balanced", remaining=0.5).tier == planner.SURVIVAL


def test_every_tier_gives_a_reason():
    for p in (plan(), plan("fast"), plan(remaining=2.4), plan(remaining=0.5)):
        assert p.reason and p.reason.startswith(p.tier)


def test_constrained_and_survival_lower_resolution():
    assert plan(remaining=2.4).resolution == 384
    survival = plan(remaining=0.5)
    assert survival.resolution == 256 and survival.checks == planner.SAFETY_ONLY


def test_queue_shrinks_budget_and_says_so():
    p = plan("best", waiting=5)
    assert p.budget_s == 5.0
    assert "5 jobs are waiting" in p.reason


def test_auto_budget_uses_device_and_complexity():
    assert plan("auto").budget_s == planner.BUDGET_SECONDS["balanced"]
    assert plan("auto", complexity=0.8).budget_s == planner.BUDGET_SECONDS["best"]
    assert plan("auto", device=planner.Device(8, 8, True)).budget_s == planner.BUDGET_SECONDS["fast"]
    assert plan("auto", device=planner.Device(2, 4, False)).budget_s == planner.BUDGET_SECONDS["fast"]


def test_save_data_caps_delivery_not_generation():
    p = plan("balanced", device=planner.Device(8, 8, True))
    assert p.resolution == 512 and p.deliver_res == 384


# --- anytime --------------------------------------------------------------


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def attempts_from(scores, clock, cost_s=1.0):
    def fn(seed):
        clock.t += cost_s
        score = scores[min(fn.n, len(scores) - 1)]
        fn.n += 1
        if score is None:
            return Attempt(seed, None, -1.0, False, rejected="blank frame")
        return Attempt(seed, object(), score, True)

    fn.n = 0
    return fn


def test_anytime_keeps_best_and_stops_at_deadline():
    clock = Clock()
    seen = []
    r = run_anytime(attempts_from([0.2, 0.9, 0.5, 0.1], clock), 3.5, 30, 1000, on_best=lambda a, n: seen.append(a.score), clock=clock, first_seed=1)
    assert r.attempts == 3
    assert r.best.score == 0.9
    assert seen == [0.2, 0.9]


def test_anytime_always_runs_one_attempt():
    clock = Clock()
    r = run_anytime(attempts_from([0.3], clock), 0.0, 30, 5000, clock=clock, first_seed=1)
    assert r.attempts == 1 and r.best is not None


def test_anytime_retries_past_deadline_only_when_empty_and_within_hard_limit():
    clock = Clock()
    r = run_anytime(attempts_from([None, None, 0.4], clock), 0.5, 30, 1000, clock=clock, first_seed=1)
    assert r.attempts == 3 and r.best.score == 0.4

    clock = Clock()
    r = run_anytime(attempts_from([None], clock), 0.5, 1.5, 1000, clock=clock, first_seed=1)
    assert r.best is None and r.stopped == "hard_deadline"


def test_anytime_cancel_stops_between_attempts():
    clock = Clock()
    r = run_anytime(attempts_from([0.3], clock), 100, 100, 1000, should_stop=lambda: clock.t >= 2, clock=clock, first_seed=1)
    assert r.attempts == 2 and r.stopped == "cancelled"


# --- classical ------------------------------------------------------------


def noise_image(seed=0, size=128):
    return np.random.default_rng(seed).integers(40, 220, (size, size, 3), dtype=np.uint8)


def casted_image():
    img = noise_image()
    img[..., 0] = np.clip(img[..., 0].astype(int) + 35, 0, 255)
    return img


def all_modes(mode):
    return {k: mode for k in classical.DEFAULT_MODES}


def ran(result, op):
    return next(o for o in result.ops if o["op"] == op)["applied"]


def test_each_op_can_be_forced_on_or_off():
    img = casted_image()
    off = classical.enhance(img, False, all_modes(classical.OFF))
    assert not any(o["applied"] for o in off.ops)
    assert np.array_equal(off.image, img)
    on = classical.enhance(img, False, all_modes(classical.ALWAYS))
    assert all(o["applied"] for o in on.ops)
    assert not np.array_equal(on.image, img)


@pytest.mark.parametrize("mode", [classical.GATED, classical.ALWAYS])
def test_gray_world_never_runs_when_prompt_names_colours(mode):
    modes = {**all_modes(classical.OFF), "gray_world": mode}
    assert ran(classical.enhance(casted_image(), False, modes), "gray_world")
    named = classical.enhance(casted_image(), True, modes)
    assert not ran(named, "gray_world")
    assert np.array_equal(named.image, casted_image())


def test_gated_gray_world_waits_for_a_cast():
    modes = {**all_modes(classical.OFF), "gray_world": classical.GATED}
    grey = np.stack([noise_image()[..., 0]] * 3, axis=-1)
    assert not ran(classical.enhance(grey, False, modes), "gray_world")


def test_blank_frame_is_severe_and_normal_is_not():
    assert classical.assess(np.zeros((64, 64, 3), np.uint8)).severe == "blank frame"
    assert classical.assess(noise_image()).severe is None


def test_saturated_blob_gets_inpainted():
    img = noise_image()
    img[40:54, 40:54] = 255
    assert classical.assess(img).blob_share > classical.SATURATED_BLOB_SHARE
    modes = {**all_modes(classical.OFF), "inpaint": classical.GATED}
    out = classical.enhance(img, False, modes)
    assert ran(out, "inpaint")
    assert out.image[47, 47].max() < 255


def test_black_blob_is_not_treated_as_a_defect():
    img = noise_image()
    img[40:54, 40:54] = 0
    assert classical.assess(img).blob_share == 0.0


def test_grid_ratio_detects_8px_checkerboard():
    img = noise_image(size=256)
    clean = classical.grid_ratio(img)
    x = np.arange(256)
    wave = (40 * np.sign(np.sin(2 * np.pi * x / 8))).astype(int)
    img2 = np.clip(img.astype(int) + wave[None, :, None], 0, 255).astype(np.uint8)
    assert classical.grid_ratio(img2) > clean * 2


def test_mode_env_rejects_unknown(monkeypatch):
    monkeypatch.setenv("KSHANA_CLASSICAL", '{"clahe":"sometimes"}')
    with pytest.raises(ValueError):
        classical.modes_from_env()
    monkeypatch.setenv("KSHANA_CLASSICAL", '{"clahe":"off"}')
    assert classical.modes_from_env()["clahe"] == "off"


# --- verifier rules -------------------------------------------------------


def det(cls, conf, box=(10, 10, 50, 50)):
    return verifier.Detection(cls, conf, np.array(box, dtype=np.float32))


def statuses(items):
    return {i["label"]: i["status"] for i in items}


BLANK = np.zeros((128, 128, 3), np.uint8)


def test_non_coco_is_not_verifiable_never_passed():
    items = verifier.build_checklist(BLANK, parser.parse("a small cabin"), [])
    assert statuses(items) == {"cabin": verifier.NOT_VERIFIABLE}


def test_presence_uses_detector_output():
    p = parser.parse("a dog and a cat")
    s = statuses(verifier.build_checklist(BLANK, p, [det("dog", 0.5)]))
    assert s == {"dog": verifier.PASS, "cat": verifier.FAIL}


def test_count_needs_point_nine_confidence():
    p = parser.parse("two dogs")
    assert statuses(verifier.build_checklist(BLANK, p, [det("dog", 0.95), det("dog", 0.5)]))["2 × dog"] == verifier.FAIL
    assert statuses(verifier.build_checklist(BLANK, p, [det("dog", 0.95), det("dog", 0.92)]))["2 × dog"] == verifier.PASS


def test_tier_without_detector_reports_not_verifiable():
    s = statuses(verifier.build_checklist(BLANK, parser.parse("two red dogs"), None))
    assert set(s.values()) == {verifier.NOT_VERIFIABLE}


def test_position_rule_and_margin():
    a, b = np.array([0, 40, 20, 60.0]), np.array([100, 40, 120, 60.0])
    assert "left of" in verifier.relative_position(a, b)
    assert "right of" in verifier.relative_position(b, a)
    top, low = np.array([40, 0, 60, 20.0]), np.array([40, 100, 60, 120.0])
    assert "above" in verifier.relative_position(top, low)
    # Overlapping almost exactly: inside the margin, so no relation at all.
    assert verifier.relative_position(np.array([0, 0, 100, 100.0]), np.array([2, 2, 102, 102.0])) == set()


def test_position_not_verifiable_without_both_objects():
    p = parser.parse("a dog to the left of a car")
    item = next(i for i in verifier.build_checklist(BLANK, p, [det("dog", 0.8)]) if i["kind"] == "position")
    assert item["status"] == verifier.NOT_VERIFIABLE


def test_depth_relation_is_not_verifiable():
    p = parser.parse("a dog behind a car")
    item = next(i for i in verifier.build_checklist(BLANK, p, [det("dog", 0.8), det("car", 0.8)]) if i["kind"] == "position")
    assert item["status"] == verifier.NOT_VERIFIABLE


@pytest.mark.parametrize(
    "rgb,name",
    [((220, 30, 30), "red"), ((30, 180, 40), "green"), ((30, 50, 220), "blue"), ((240, 240, 240), "white"), ((10, 10, 10), "black"), ((230, 220, 30), "yellow")],
)
def test_colour_shares_on_solid_patches(rgb, name):
    patch = np.full((32, 32, 3), rgb, np.uint8)
    assert verifier.colour_shares(patch)[name] == 1.0


def test_colour_check_inside_box():
    img = np.zeros((128, 128, 3), np.uint8)
    img[10:50, 10:50] = (220, 30, 30)
    p = parser.parse("a red dog")
    assert statuses(verifier.build_checklist(img, p, [det("dog", 0.8)]))["red dog"] == verifier.PASS
    p = parser.parse("a blue dog")
    assert statuses(verifier.build_checklist(img, p, [det("dog", 0.8)]))["blue dog"] == verifier.FAIL


def test_score_prefers_passing_and_floors_severe():
    good = [{"status": verifier.PASS}, {"status": verifier.PASS}]
    bad = [{"status": verifier.FAIL}, {"status": verifier.PASS}]
    assert verifier.score_candidate(good, 0.3, None) > verifier.score_candidate(bad, 0.3, None)
    assert verifier.score_candidate(good, 0.3, "blank frame") == -1.0


# --- safety ---------------------------------------------------------------


@pytest.mark.parametrize("text", ["a nude person", "N U D E woman", "p0rn", "beheaded man", "a swastika flag", "a child in a bikini"])
def test_blocklist_refuses(text):
    assert safety.check_prompt(text) is not None


@pytest.mark.parametrize(
    "text",
    ["a red apple on a table", "a child flying a kite", "a baby elephant", "a nudibranch in the sea", "essex countryside", "a chicken breast on a plate"][:5],
)
def test_blocklist_allows_ordinary_prompts(text):
    assert safety.check_prompt(text) is None


class FakeClip:
    def __init__(self):
        self.rng = np.random.default_rng(0)

    def embed_text(self, texts):
        v = self.rng.normal(size=(len(texts), 16))
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_concept_check_flags_only_near_unsafe_embeddings():
    check = safety.ConceptCheck(FakeClip())
    assert check.flagged(check.unsafe[0])
    assert not check.flagged(check.safe[0])


def test_label_is_burned_in_visibly_and_does_not_touch_input():
    img = np.full((512, 512, 3), 128, np.uint8)
    stamped = safety.stamp_label(img)
    assert img.mean() == 128
    assert np.any(stamped[-40:, -160:] != 128)
    assert np.array_equal(stamped[:400], img[:400])
    assert safety.stamp_label(np.full((256, 256, 3), 128, np.uint8)).shape == (256, 256, 3)


# --- low memory -----------------------------------------------------------


def test_on_demand_session_exists_only_during_a_run(monkeypatch):
    live = []

    class FakeSession:
        def __init__(self):
            live.append(self)

        def run(self, outputs, feed):
            return ["ran", feed]

    from kshana import ort as kort

    monkeypatch.setattr(kort, "session", lambda path, threads=None: FakeSession())
    model = kort.OnDemand("x.onnx", 2)
    assert live == []
    assert model.run(None, {"a": 1})[0] == "ran"
    assert len(live) == 1


def test_low_memory_setting_picks_the_session_type(monkeypatch):
    from kshana import ort as kort

    monkeypatch.setattr(kort, "session", lambda path, threads=None: "resident")
    monkeypatch.delenv("KSHANA_LOW_MEMORY", raising=False)
    assert kort.open_model("x.onnx") == "resident"
    monkeypatch.setenv("KSHANA_LOW_MEMORY", "1")
    assert isinstance(kort.open_model("x.onnx"), kort.OnDemand)


def test_clip_text_batches_into_one_load():
    calls = []

    class FakeText:
        def run(self, outputs, feed):
            calls.append(feed["input_ids"].shape)
            return [np.ones((feed["input_ids"].shape[0], 8), np.float32)]

    class FakeTok:
        def encode(self, text, add_special_tokens=True):
            return type("E", (), {"ids": [49406, 5, 49407]})()

    clip = verifier.Clip.__new__(verifier.Clip)
    clip.text, clip.tok = FakeText(), FakeTok()
    out = clip.embed_text(["a", "b", "c"])
    assert calls == [(3, 77)]
    assert out.shape == (3, 8)
    assert abs(float(np.linalg.norm(out[0])) - 1.0) < 1e-5


# --- weights download -----------------------------------------------------


@pytest.fixture
def weights_server(tmp_path):
    import functools
    import http.server
    import threading

    src = tmp_path / "published"
    (src / "tokenizer").mkdir(parents=True)
    for name in weights.MANIFEST:
        (src / name).write_bytes(name.encode() * 10)

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(src)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", src
    server.shutdown()


def test_weights_download_fills_a_fresh_folder(tmp_path, monkeypatch, weights_server):
    url, src = weights_server
    monkeypatch.setenv("KSHANA_WEIGHTS_URL", url)
    dst = tmp_path / "models"
    assert weights.missing(dst) == weights.MANIFEST
    seen = []
    weights.ensure(dst, lambda name, total: seen.append(total))
    assert weights.missing(dst) == []
    assert (dst / "tokenizer/meta.json").read_bytes() == (src / "tokenizer/meta.json").read_bytes()
    assert seen == sorted(seen) and seen[-1] > 0
    assert not list(dst.rglob("*.part"))


def test_weights_download_keeps_existing_files(tmp_path, monkeypatch, weights_server):
    url, _ = weights_server
    monkeypatch.setenv("KSHANA_WEIGHTS_URL", url)
    dst = tmp_path / "models"
    dst.mkdir()
    (dst / "yolox.onnx").write_bytes(b"already here")
    weights.ensure(dst)
    assert (dst / "yolox.onnx").read_bytes() == b"already here"


def test_failed_download_leaves_nothing_that_looks_finished(tmp_path, monkeypatch, weights_server):
    url, src = weights_server
    (src / "unet.onnx.data").unlink()
    monkeypatch.setenv("KSHANA_WEIGHTS_URL", url)
    dst = tmp_path / "models"
    with pytest.raises(Exception):
        weights.ensure(dst)
    assert "unet.onnx.data" in weights.missing(dst)


def test_upload_script_lists_the_download_manifest():
    # The two lists must not drift: this is what makes a published repo bootable.
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "optimize" / "upload_weights.py"
    spec = importlib.util.spec_from_file_location("upload_weights", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.MANIFEST is weights.MANIFEST
