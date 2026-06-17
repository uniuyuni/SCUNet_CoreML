from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import requests

from .export_coreml import export_coreml


SCUNET_REPO = "https://github.com/cszn/SCUNet.git"
CHECKPOINT_URL = "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth"


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def clone_scunet(root: Path) -> Path:
    dst = root / "third_party" / "SCUNet"
    if dst.exists():
        print(f"SCUNet repo already exists: {dst}")
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", "1", SCUNET_REPO, str(dst)])
    return dst


def download_checkpoint(root: Path, force: bool = False) -> Path:
    out = root / "checkpoints" / "scunet_color_real_psnr.pth"
    if out.exists() and not force:
        print(f"checkpoint already exists: {out}")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {CHECKPOINT_URL} -> {out}")
    with requests.get(CHECKPOINT_URL, stream=True, timeout=120) as response:
        response.raise_for_status()
        with out.open("wb") as f:
            shutil.copyfileobj(response.raw, f)
    return out


def export_models(root: Path, weights: Path, force: bool = False) -> dict[str, str]:
    out = root / "models" / "scunet_color_real_psnr_448_fp16.mlpackage"
    if out.exists() and not force:
        print(f"Core ML model already exists: {out}")
    else:
        export_coreml(weights, out, tile=448, convert_to="mlprogram", precision="float16", compute_units="cpu_and_gpu")
    return {"model": str(out), "checkpoint": str(weights), "scunet_repo": str(root / "third_party" / "SCUNet")}


def write_manifest(root: Path, assets: dict[str, str]) -> None:
    manifest = {"scunet_repo": SCUNET_REPO, "checkpoint_url": CHECKPOINT_URL, "tile": 448, "overlap": 64, "assets": assets}
    path = root / "models" / "assets.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Set up SCUNet CoreML assets.")
    ap.add_argument("--root", default=".")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-clone", action="store_true")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not args.skip_clone:
        clone_scunet(root)
    weights = root / "checkpoints" / "scunet_color_real_psnr.pth"
    if not args.skip_download:
        weights = download_checkpoint(root, force=args.force)
    if not weights.exists():
        raise FileNotFoundError(f"checkpoint is missing: {weights}")
    if not args.skip_export:
        assets = export_models(root, weights, force=args.force)
        write_manifest(root, assets)


if __name__ == "__main__":
    main()
