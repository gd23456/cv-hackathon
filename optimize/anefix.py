# Neural Engine SiLU fix: the ANE's SiLU unit is low-accuracy, and the one-step
# x0 formula divides by 0.0683, amplifying UNet error about 14.6x.
# SiLU is rebuilt from add/mul/clip, which the ANE runs accurately.
import torch


class PadeSiLU(torch.nn.Module):
    """SiLU(x) = 0.5*x*(1 + tanh(x/2)), tanh by a [7/6] Pade ratio on x clipped to +-L.

    Constants are normalised by 135135 so FP16 intermediates stay in range.
    """

    A = (17325 / 135135, 378 / 135135, 1 / 135135)
    B = (62370 / 135135, 3150 / 135135, 28 / 135135)

    def __init__(self, clip_at: float = 12.0):
        super().__init__()
        self.clip_at = float(clip_at)

    def forward(self, x):
        u = torch.clamp(x, -self.clip_at, self.clip_at)
        y = 0.5 * u
        t = y * y
        a, b = self.A, self.B
        num = y * (1.0 + t * (a[0] + t * (a[1] + t * a[2])))
        den = 1.0 + t * (b[0] + t * (b[1] + t * b[2]))
        return 0.5 * x * (1.0 + num / den)


def patch_silu(module: torch.nn.Module, clip_at: float = 12.0) -> int:
    """Replace every nn.SiLU child in place; returns how many were replaced."""
    replaced = 0
    for parent in module.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, torch.nn.SiLU):
                setattr(parent, name, PadeSiLU(clip_at))
                replaced += 1
    return replaced
