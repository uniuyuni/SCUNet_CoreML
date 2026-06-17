from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .model import build_scunet_color_real_psnr


def compute_unit(name: str):
    import coremltools as ct

    return {
        "all": ct.ComputeUnit.ALL,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
    }[name]


def load_scunet(weights: str | Path, tile: int = 448) -> torch.nn.Module:
    state = torch.load(weights, map_location="cpu", weights_only=False)
    model = build_scunet_color_real_psnr(tile=tile)
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [k for k in unexpected if not k.endswith("attn_bias") and not k.endswith("relative_index")]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model


def export_coreml(
    weights: str | Path,
    output: str | Path,
    *,
    tile: int = 448,
    convert_to: str = "mlprogram",
    precision: str = "float16",
    compute_units: str = "cpu_and_gpu",
) -> Path:
    import coremltools as ct

    model = load_scunet(weights, tile=tile)
    example = torch.rand(1, 3, tile, tile, dtype=torch.float32)
    with torch.inference_mode():
        traced = torch.jit.trace(model, example, strict=False)
        traced = torch.jit.freeze(traced.eval())
        max_diff = float((model(example) - traced(example)).abs().max().item())
    print(f"trace max abs diff: {max_diff:.8f}")

    kwargs = {
        "convert_to": convert_to,
        "inputs": [ct.TensorType(name="input", shape=example.shape, dtype=np.float32)],
        "outputs": [ct.TensorType(name="output", dtype=np.float32)],
        "compute_units": compute_unit(compute_units),
        "minimum_deployment_target": ct.target.macOS13,
    }
    if convert_to == "mlprogram":
        kwargs["compute_precision"] = ct.precision.FLOAT16 if precision == "float16" else ct.precision.FLOAT32

    mlmodel = ct.convert(traced, **kwargs)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out))
    print(f"wrote {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Export SCUNet color real PSNR checkpoint to Core ML.")
    ap.add_argument("--weights", default="checkpoints/scunet_color_real_psnr.pth")
    ap.add_argument("--output", default="models/scunet_color_real_psnr_448_fp16.mlpackage")
    ap.add_argument("--tile", type=int, default=448)
    ap.add_argument("--convert-to", choices=["mlprogram", "neuralnetwork"], default="mlprogram")
    ap.add_argument("--precision", choices=["float32", "float16"], default="float16")
    ap.add_argument("--compute-units", choices=["all", "cpu_only", "cpu_and_gpu", "cpu_and_ne"], default="cpu_and_gpu")
    args = ap.parse_args()
    export_coreml(
        args.weights,
        args.output,
        tile=args.tile,
        convert_to=args.convert_to,
        precision=args.precision,
        compute_units=args.compute_units,
    )


if __name__ == "__main__":
    main()
