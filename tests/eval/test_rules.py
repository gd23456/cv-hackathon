"""Fast unit tests for statistics, scorer rules, splits, and guards. No models needed.

    python -m pytest tests/eval/test_rules.py -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "optimize"))
import common  # noqa: E402
import geneval_score as gs  # noqa: E402
import metrics  # noqa: E402


def no_color(_boxes, _name):
    raise AssertionError("colour classifier should not run")


def test_relative_position_matches_geneval_rule():
    a, b = [0, 0, 10, 10], [100, 0, 110, 10]
    assert gs.relative_position(a, b) == {"left of"}
    assert gs.relative_position(b, a) == {"right of"}
    assert gs.relative_position([0, 0, 10, 10], [0, 100, 10, 110]) == {"above"}
    assert gs.relative_position([0, 100, 10, 110], [0, 0, 10, 10]) == {"below"}
    assert gs.relative_position([0, 0, 10, 10], [0.5, 0.5, 10.5, 10.5]) == set()  # inside the 0.1 margin


def test_position_check():
    row = {"tag": "position", "include": [{"class": "dog", "count": 1}, {"class": "cat", "count": 1, "position": ["right of", 0]}]}
    good = {"dog": [[0, 0, 10, 10]], "cat": [[100, 0, 110, 10]]}
    bad = {"dog": [[100, 0, 110, 10]], "cat": [[0, 0, 10, 10]]}
    assert gs.evaluate(row, good, no_color)[0]["position"] is True
    assert gs.evaluate(row, bad, no_color)[0]["position"] is False


def test_count_and_exclude():
    row = {"tag": "counting", "include": [{"class": "clock", "count": 2}], "exclude": [{"class": "clock", "count": 3}]}
    box = [0, 0, 1, 1]
    assert gs.evaluate(row, {"clock": [box, box]}, no_color)[0]["count"] is True
    assert gs.evaluate(row, {"clock": [box]}, no_color)[0]["count"] is False
    assert gs.evaluate(row, {"clock": [box, box, box]}, no_color)[0]["count"] is False


def test_absent_object_fails_presence():
    row = {"tag": "two_object", "include": [{"class": "dog", "count": 1}, {"class": "cat", "count": 1}]}
    checks, _ = gs.evaluate(row, {"dog": [[0, 0, 1, 1]]}, no_color)
    assert checks["presence"] is False


def test_color_check_uses_classifier():
    row = {"tag": "colors", "include": [{"class": "car", "count": 1, "color": "blue"}]}
    assert gs.evaluate(row, {"car": [[0, 0, 1, 1]]}, lambda b, n: ["blue"])[0]["color"] is True
    assert gs.evaluate(row, {"car": [[0, 0, 1, 1]]}, lambda b, n: ["red"])[0]["color"] is False
    assert gs.evaluate(row, {}, no_color)[0]["color"] is False


def test_paired_bootstrap_detects_shift_and_null():
    rng = np.random.default_rng(1)
    a = rng.normal(30, 5, 1000)
    assert metrics.paired_bootstrap(a + 1, a)["excludes_zero"]
    assert metrics.paired_bootstrap(a, a + rng.normal(0, 1, 1000))["excludes_zero"] is False


def test_cmmd_zero_for_identical_and_positive_for_shifted():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(300, 16))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y = x + 0.5
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    assert abs(metrics.cmmd_x1000(x, x)) < 1e-6
    assert metrics.cmmd_x1000(x, y) > 0


def test_fid_refuses_below_5k():
    with pytest.raises(ValueError):
        metrics.fid(np.zeros((1000, 2048)), np.zeros((5000, 2048)))


def test_test_split_guard():
    with pytest.raises(SystemExit):
        common.guard_test_split("complex_test", False)
    common.guard_test_split("complex_test", True)
    common.guard_test_split("complex_dev", False)


def test_splits_disjoint_and_stable():
    dev = common.read_split("complex_dev")
    test = common.read_split("complex_test")
    assert not {r["id"] for r in dev} & {r["id"] for r in test}
    assert len(dev) + len(test) == 553 + 300
    assert 190 <= len(dev) <= 210
    coco5k = {r["id"] for r in common.read_split("coco_5k")}
    assert {r["id"] for r in common.read_split("coco_1k")} <= coco5k
    assert len(coco5k) == 5000


def test_seed_latent_is_stable_and_matches_spike_recipe():
    import torch

    a = common.latent_for_seed(42)
    b = torch.randn((1, 4, 64, 64), generator=torch.Generator("cpu").manual_seed(42), dtype=torch.float32)
    assert torch.equal(a, b)
    assert a.shape == (1, 4, 64, 64)


def test_pade_silu_accuracy():
    import torch
    from anefix import PadeSiLU

    x = torch.linspace(-6, 6, 4001)
    assert (PadeSiLU()(x) - torch.nn.functional.silu(x)).abs().max() < 2e-3


def test_csv_upsert_replaces_key(tmp_path):
    p = tmp_path / "t.csv"
    metrics.upsert_csv(p, [{"a": 1, "v": 1}, {"a": 2, "v": 2}], ("a",))
    metrics.upsert_csv(p, [{"a": 1, "v": 9}], ("a",))
    text = p.read_text()
    assert "1,9" in text and "2,2" in text and "1,1" not in text


def test_budget_guard_counts_in_flight_workers_at_worst_case():
    from budget import BudgetGuard

    g = BudgetGuard(budget_usd=12.0, spent_usd=0.0, worst_chunk_usd=1.0, startup_usd=0.5)
    assert g.may_dispatch(in_flight=0, new_container=True)
    g.charge(10.6)
    assert g.may_dispatch(in_flight=0, new_container=False)  # 10.6 + 1 = 11.6
    assert not g.may_dispatch(in_flight=0, new_container=True)  # 10.6 + 1 + 0.5 = 12.1
    assert not g.may_dispatch(in_flight=1, new_container=False)  # 10.6 + 2 = 12.6


def test_budget_guard_carries_earlier_spend():
    from budget import BudgetGuard

    assert not BudgetGuard(12.0, 11.5, 1.0, 0.5).may_dispatch(0, False)


def test_modal_backstop_covers_budget():
    from budget import Rates, worker_rate

    rate = worker_rate(Rates(), 4.0, 16.0)
    assert 0.0002 < rate < 0.001


def test_best_index_uses_only_first_n():
    from candidates import best_index, seed_for

    sel = np.array([[0.1, 0.9, 0.5, 2.0], [0.7, 0.2, 0.3, 0.1]])
    assert best_index(sel, 1).tolist() == [0, 0]
    assert best_index(sel, 2).tolist() == [1, 0]
    assert best_index(sel, 4).tolist() == [3, 0]
    assert seed_for(7, 0) == 7  # candidate 0 is the single-image run


def test_eval_md_line_one_and_order(tmp_path, monkeypatch):
    import write_eval_md as w

    monkeypatch.setattr(w, "RESULTS", tmp_path)
    monkeypatch.setattr(w, "COREML", tmp_path)
    w.main()
    text = (tmp_path / "eval_P3_T78.md").read_text()
    assert text.split("\n")[0] == "1RUA24CSE0146,1RUA24CSE0201,1RUA24CSE0139,1RUA24CSE0196"
    headings = [line for line in text.split("\n") if line.startswith("## ")]
    assert [h.split(". ", 1)[1] for h in headings] == ["CLIP similarity", "FID", "Image generation latency", "Peak memory consumption", "Model size", "Quality versus computational cost"]
    assert "TODO" in text


def test_config_info_and_classical_names():
    from configs import config_info

    info = config_info("S1-A+clahe=always")
    assert info["classical"] == "clahe=always" and "clahe=always" in info["precision"]
    assert "classical" not in config_info("S1-A")
