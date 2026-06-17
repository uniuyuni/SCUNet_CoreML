from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="Install SCUNet CoreML helper into Platypus.")
    ap.add_argument("--platypus-root", default=str(root.parent))
    args = ap.parse_args()

    helpers_dir = Path(args.platypus_root).expanduser().resolve() / "helpers"
    if not helpers_dir.exists():
        raise FileNotFoundError(f"Platypus helpers directory is missing: {helpers_dir}")
    src = root / "platypus_helpers" / "scunet_coreml_helper.py"
    dst = helpers_dir / "scunet_coreml_helper.py"
    shutil.copy2(src, dst)
    print(f"installed {dst}")


if __name__ == "__main__":
    main()
