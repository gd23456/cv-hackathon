#!/usr/bin/env python
"""Intermediate images for the report: one generated image through each classical step."""
from __future__ import annotations

import cv2
import numpy as np

from kshana import classical

rgb = cv2.imread("web/tunnel/5.jpg")[..., ::-1].copy()
grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
edges = cv2.magnitude(cv2.Sobel(grey, cv2.CV_32F, 1, 0), cv2.Sobel(grey, cv2.CV_32F, 0, 1))
edges = np.clip(edges / max(edges.max(), 1) * 255 * 2, 0, 255).astype(np.uint8)
steps = [
    ("generated", rgb), ("edge map", cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)),
    ("CLAHE", classical.clahe(rgb)), ("edge-masked unsharp", classical.edge_masked_unsharp(rgb)),
]
tiles = []
for name, image in steps:
    tile = image.copy()
    cv2.rectangle(tile, (0, 0), (tile.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(tile, name, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    tiles.append(tile)
cv2.imwrite("docs/figures/classical_steps.png", np.hstack(tiles)[..., ::-1])
