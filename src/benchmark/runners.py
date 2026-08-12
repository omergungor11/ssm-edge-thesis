"""Yığın-agnostik runner arayüzü.

Her runner: prepare(x) → hazırlık, run_once() → tek ileri geçiş, sync() → tamamlanma
garantisi, stats() → runner'a özgü ek bilgi (yükleme süresi, bellek tepe noktası...).
"""
from __future__ import annotations

import time

import numpy as np
import torch


class TorchRunner:
    def __init__(self, model: torch.nn.Module, device: str = "cpu", compile_mode: str | None = None):
        self.device = device
        self.label = f"torch_{device}" + (f"_compile-{compile_mode}" if compile_mode else "")
        t0 = time.perf_counter()
        self.model = model.eval().to(device)
        if compile_mode:
            self.model = torch.compile(self.model, mode=compile_mode)
        self._prep_s = round(time.perf_counter() - t0, 3)
        self._x: torch.Tensor | None = None

    def prepare(self, x: np.ndarray) -> None:
        self._x = torch.from_numpy(x).to(self.device)
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def run_once(self):
        with torch.no_grad():
            return self.model(self._x)

    def sync(self) -> None:
        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device == "cuda":
            torch.cuda.synchronize()

    def stats(self) -> dict:
        d: dict = {"prep_s": self._prep_s}
        if self.device == "mps":
            d["mps_peak_bytes"] = torch.mps.driver_allocated_memory()
        return d


class OrtRunner:
    def __init__(self, onnx_path: str, providers: list[str] | None = None, opt_level: str = "all"):
        import onnxruntime as ort

        self.label = "ort_" + "+".join(p.replace("ExecutionProvider", "") for p in (providers or ["CPUExecutionProvider"]))
        so = ort.SessionOptions()
        so.graph_optimization_level = {
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "none": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        }[opt_level]
        t0 = time.perf_counter()
        self.sess = ort.InferenceSession(onnx_path, so, providers=providers or ["CPUExecutionProvider"])
        self._load_s = round(time.perf_counter() - t0, 3)
        self._active = self.sess.get_providers()
        self._feed: dict | None = None

    def prepare(self, x: np.ndarray) -> None:
        self._feed = {self.sess.get_inputs()[0].name: x}

    def run_once(self):
        return self.sess.run(None, self._feed)

    def sync(self) -> None:  # ORT run() senkron döner
        pass

    def stats(self) -> dict:
        return {"load_s": self._load_s, "active_providers": self._active}


class CoreMLRunner:
    def __init__(self, mlpackage_path: str, compute_units: str = "ALL"):
        import coremltools as ct

        self.label = f"coreml_{compute_units.lower()}"
        cu = getattr(ct.ComputeUnit, compute_units)
        t0 = time.perf_counter()
        self.model = ct.models.MLModel(mlpackage_path, compute_units=cu)
        self._load_s = round(time.perf_counter() - t0, 3)
        self._feed: dict | None = None

    def prepare(self, x: np.ndarray) -> None:
        self._feed = {self.model.get_spec().description.input[0].name: x}

    def run_once(self):
        return self.model.predict(self._feed)

    def sync(self) -> None:  # predict() senkron döner
        pass

    def stats(self) -> dict:
        return {"load_s": self._load_s}
