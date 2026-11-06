"""Shared ONNX Runtime session setup."""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import onnxruntime as ort


def session(path: Path, threads: int | None = None) -> ort.InferenceSession:
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = int(threads)
    so.inter_op_num_threads = 1
    # No arena: each tensor is freed as soon as its last consumer has run, which keeps the peak low.
    so.enable_cpu_mem_arena = False
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def give_back_freed_memory() -> None:
    """The allocator keeps freed pages for reuse, and they count against the host's limit until returned."""
    if sys.platform == "darwin":
        ctypes.CDLL(None).malloc_zone_pressure_relief(None, 0)
    elif sys.platform.startswith("linux"):
        ctypes.CDLL("libc.so.6").malloc_trim(0)


class OnDemand:
    """A session that exists only while it runs, so models never share memory with each other.
    Costs a load per run: about 0.3 s for the UNet on an M4, since its weights are memory-mapped."""

    def __init__(self, path: Path, threads: int | None = None):
        self.path, self.threads = path, threads

    def run(self, outputs, feed):
        result = session(self.path, self.threads).run(outputs, feed)
        give_back_freed_memory()
        return result


def low_memory() -> bool:
    return os.environ.get("KSHANA_LOW_MEMORY") == "1"


def open_model(path: Path, threads: int | None = None):
    return OnDemand(path, threads) if low_memory() else session(path, threads)
