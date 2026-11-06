"""My in-memory Inception path must equal clean-fid's file path. Reference sanity: real vs real is near zero.

    python -m pytest tests/eval/test_inception_matches_cleanfid.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
from common import DATA, FEATURES  # noqa: E402
from inception import WEIGHTS_DIR, InceptionFeatures  # noqa: E402

has_weights = (WEIGHTS_DIR / "inception-2015-12-05.pt").exists()
REF = FEATURES / "reference/coco5k.npz"


@pytest.mark.skipif(not has_weights, reason="Inception file not fetched")
def test_features_equal_cleanfid_file_path(tmp_path):
    import torch
    from cleanfid import fid

    paths = []
    for i, src in enumerate(sorted((DATA / "coco/val2017").glob("*.jpg"))[:4]):
        with Image.open(src) as im:
            im.convert("RGB").resize((512, 512)).save(tmp_path / f"{i}.png")
        paths.append(str(tmp_path / f"{i}.png"))
    ours = InceptionFeatures()
    arrays = [np.array(Image.open(p).convert("RGB")) for p in paths]
    theirs = fid.get_files_features(paths, model=lambda x: ours.model(x), num_workers=0, batch_size=4, device=torch.device("cpu"), mode="clean", verbose=False)
    assert np.allclose(ours(arrays), theirs, atol=1e-3)


@pytest.mark.skipif(not REF.exists(), reason="reference features not computed")
def test_real_vs_real_is_near_zero():
    from metrics import cmmd_x1000, kid_x1000

    with np.load(REF) as f:
        keys = f.files
        inc = f["inception"] if "inception" in keys else None
        emb = f["emb_l14"]
    order = np.random.default_rng(0).permutation(len(emb))
    a, b = order[:2500], order[2500:]
    assert abs(cmmd_x1000(emb[a], emb[b])) < 5.0
    if inc is not None:
        assert abs(kid_x1000(inc[a], inc[b])) < 2.0
