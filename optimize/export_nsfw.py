#!/usr/bin/env python
"""Export Marqo/nsfw-image-detection-384 (Apache-2.0) for the output safety gate.

Input: 1x3x384x384 RGB float in 0...255. Output: probabilities [NSFW, SFW].
    python optimize/export_nsfw.py --onnx server/models/nsfw.onnx --coreml data/coreml_safety/nsfw_marqo.mlpackage
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import timm
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

REPO = "Marqo/nsfw-image-detection-384"


class Wrapped(torch.nn.Module):
    """Builds preprocessing and softmax into the model, so every platform uses the same ones."""

    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return torch.softmax(self.model((x / 255.0 - self.mean) / self.std), dim=-1)


def load() -> torch.nn.Module:
    config = json.load(open(hf_hub_download(REPO, "config.json")))
    assert config["label_names"] == ["NSFW", "SFW"], "label order changed"
    model = timm.create_model(config["architecture"], pretrained=False, num_classes=config["num_classes"])
    model.load_state_dict(load_file(hf_hub_download(REPO, "model.safetensors")))
    pretrained = config["pretrained_cfg"]
    return Wrapped(model.eval(), pretrained["mean"], pretrained["std"]).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path)
    ap.add_argument("--coreml", type=Path)
    args = ap.parse_args()

    model = load()
    example = torch.rand(1, 3, 384, 384) * 255
    with torch.no_grad():
        reference = model(example).numpy()

    if args.onnx:
        args.onnx.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            model, (example,), str(args.onnx), input_names=["image"], output_names=["probs"],
            opset_version=17, dynamo=False,
        )
        import onnxruntime as ort

        out = ort.InferenceSession(str(args.onnx)).run(None, {"image": example.numpy().astype(np.float32)})[0]
        print("onnx max diff vs torch", float(np.abs(out - reference).max()))

    if args.coreml:
        import coremltools as ct

        with torch.no_grad():
            traced = torch.jit.trace(model, example)
        converted = ct.convert(
            traced, inputs=[ct.TensorType(name="image", shape=example.shape)],
            outputs=[ct.TensorType(name="probs")], minimum_deployment_target=ct.target.iOS17,
            compute_precision=ct.precision.FLOAT16, convert_to="mlprogram",
        )
        converted.save(str(args.coreml))
        out = converted.predict({"image": example.numpy().astype(np.float32)})["probs"]
        print("coreml max diff vs torch", float(np.abs(out - reference).max()))


if __name__ == "__main__":
    main()
