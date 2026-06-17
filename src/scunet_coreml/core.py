from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np


def _tile_starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, max(1, length - tile + 1), stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def _pad_reflect_to_tile(x: np.ndarray, tile: int) -> tuple[np.ndarray, int, int]:
    h, w = x.shape[:2]
    pad_h = max(0, tile - h) if h < tile else 0
    pad_w = max(0, tile - w) if w < tile else 0
    if pad_h or pad_w:
        x = np.pad(x, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    return x, pad_h, pad_w


def _make_weight(tile: int, overlap: int) -> np.ndarray:
    if overlap <= 0:
        return np.ones((tile, tile, 1), dtype=np.float32)
    edge = min(overlap, tile // 2)
    ramp = np.ones(tile, dtype=np.float32)
    vals = np.linspace(0.0, 1.0, edge + 2, dtype=np.float32)[1:-1]
    ramp[:edge] = vals
    ramp[-edge:] = vals[::-1]
    return (ramp[:, None] * ramp[None, :])[..., None].astype(np.float32)


def compute_unit(name: str):
    import coremltools as ct

    return {
        "all": ct.ComputeUnit.ALL,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
    }[name]


@dataclass
class DenoiseConfig:
    model_path: str | Path
    compute_units: str = "cpu_and_gpu"
    tile: int = 448
    overlap: int = 64
    progress_every: int = 4
    progress_callback: Callable[[int, int], None] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SCUNetCoreML:
    def __init__(self, config: DenoiseConfig):
        import coremltools as ct

        self.config = config
        self.model = ct.models.MLModel(str(config.model_path), compute_units=compute_unit(config.compute_units))

    def denoise(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        cfg = self.config
        src = np.ascontiguousarray(np.asarray(image, dtype=np.float32))
        src = np.nan_to_num(src, nan=0.0, posinf=1.0, neginf=0.0)
        if src.ndim != 3 or src.shape[-1] < 3:
            raise ValueError(f"expected HxWx3 image, got {src.shape}")
        rgb = src[..., :3]
        alpha = src[..., 3:] if src.shape[-1] > 3 else None

        padded, _, _ = _pad_reflect_to_tile(rgb, cfg.tile)
        h, w = rgb.shape[:2]
        stride = cfg.tile - cfg.overlap
        ys = _tile_starts(padded.shape[0], cfg.tile, stride)
        xs = _tile_starts(padded.shape[1], cfg.tile, stride)
        coords = [(y, x) for y in ys for x in xs]
        total = len(coords)
        weight = _make_weight(cfg.tile, cfg.overlap)
        out_acc = np.zeros_like(padded, dtype=np.float32)
        weight_acc = np.zeros((*padded.shape[:2], 1), dtype=np.float32)
        start = time.perf_counter()
        predict_time = 0.0
        for done, (y, x) in enumerate(coords, start=1):
            patch = padded[y : y + cfg.tile, x : x + cfg.tile].transpose(2, 0, 1)[None].astype(np.float32, copy=False)
            t0 = time.perf_counter()
            pred = self.model.predict({"input": patch})
            predict_time += time.perf_counter() - t0
            out = np.asarray(pred["output"], dtype=np.float32)[0].transpose(1, 2, 0)
            out_acc[y : y + cfg.tile, x : x + cfg.tile] += out * weight
            weight_acc[y : y + cfg.tile, x : x + cfg.tile] += weight
            if cfg.progress_callback is not None and (done == total or done % max(1, cfg.progress_every) == 0):
                cfg.progress_callback(done, total)
        out = out_acc / np.maximum(weight_acc, 1e-8)
        out = out[:h, :w]
        if alpha is not None:
            out = np.concatenate([out, alpha], axis=2)
        elapsed = time.perf_counter() - start
        return out.astype(np.float32, copy=False), {
            "tiles": total,
            "elapsed_sec": elapsed,
            "ms_per_tile": elapsed / max(1, total) * 1000.0,
            "predict_ms_per_tile": predict_time / max(1, total) * 1000.0,
            "config": {
                "model_path": str(cfg.model_path),
                "compute_units": cfg.compute_units,
                "tile": cfg.tile,
                "overlap": cfg.overlap,
            },
        }


def denoise_array(image: np.ndarray, config: DenoiseConfig) -> tuple[np.ndarray, dict[str, Any]]:
    return SCUNetCoreML(config).denoise(image)
