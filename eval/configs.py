"""Every evaluated generator: what it is and where it runs. Labels travel with every number."""

M4_ANE = {"hardware": "Apple M4", "engine": "Core ML (coremltools 9.0), CPU_AND_NE"}
M4_MPS = {"hardware": "Apple M4", "engine": "PyTorch 2.7 MPS, fp32 compute"}

CONFIGS = {
    "S1-A": {"kind": "coreml", "model": "sdxs-dreamshaper", "variant": "A", "steps": 1, "decoder": "TAESD",
             "precision": "8-bit palette, SiLU fix", **M4_ANE},
    "S1-FP16": {"kind": "coreml", "model": "sdxs-dreamshaper", "variant": "FP16", "steps": 1, "decoder": "TAESD",
                "precision": "FP16, SiLU fix", **M4_ANE},
    "S1-PAL6": {"kind": "coreml", "model": "sdxs-dreamshaper", "variant": "PAL6", "steps": 1, "decoder": "TAESD",
                "precision": "6-bit palette, SiLU fix", **M4_ANE},
    "S1-STOCK": {"kind": "coreml", "model": "sdxs-dreamshaper", "variant": "STOCK", "steps": 1, "decoder": "TAESD",
                 "precision": "FP16, no SiLU fix", **M4_ANE},
    "S0.9-FP16": {"kind": "coreml", "model": "sdxs-0.9", "variant": "FP16", "steps": 1, "decoder": "TAESD",
                  "precision": "FP16, SiLU fix", **M4_ANE},
    "B1": {"kind": "diffusers", "repo": "stabilityai/sd-turbo", "steps": 1, "cfg": 0.0, "decoder": "full VAE",
           "precision": "fp16 weights, fp32 compute", **M4_MPS},
    "B0": {"kind": "diffusers", "repo": "stable-diffusion-v1-5/stable-diffusion-v1-5", "steps": 25, "cfg": 7.5, "decoder": "full VAE",
           "precision": "fp16 weights, fp32 compute", **M4_MPS},
}
CONFIGS["S1-VAE"] = {"kind": "coreml_vae", "model": "sdxs-dreamshaper", "variant": "A", "steps": 1, "decoder": "full VAE (MPS fp32)",
                     "precision": "8-bit palette UNet and text, SiLU fix", "hardware": "Apple M4", "engine": "Core ML CPU_AND_NE + PyTorch MPS decoder"}
CONFIGS["S1-FP32"] = {"kind": "torch_sdxs", "model": "sdxs-dreamshaper", "steps": 1, "decoder": "TAESD", "precision": "fp32, no fix needed", **M4_MPS}
# SD-Turbo at C steps for E1: same weights, one UNet call per step.
for _c in (2, 4, 8):
    CONFIGS[f"B1-{_c}step"] = {**CONFIGS["B1"], "steps": _c}
CONFIGS["B0-T4"] = {**CONFIGS["B0"], "hardware": "Modal NVIDIA T4", "engine": "PyTorch 2.7 CUDA, fp32 compute"}
CONFIGS["B1-T4"] = {**CONFIGS["B1"], "hardware": "Modal NVIDIA T4", "engine": "PyTorch 2.7 CUDA, fp32 compute"}


def config_info(name: str) -> dict:
    """'S1-A+clahe=always' is S1-A with one classical op forced; the rest stay off."""
    base, _, op = name.partition("+")
    info = dict(CONFIGS[base])
    if op:
        info["classical"] = op
        info["precision"] = f"{info['precision']}; classical {op}"
    return info
