"""Download a MediaPipe PoseLandmarker .task model into models/.

Usage:
    python scripts/download_model.py            # downloads 'full'
    python scripts/download_model.py --variant lite
    python scripts/download_model.py --variant heavy
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

MODELS = {
    "lite": "pose_landmarker_lite",
    "full": "pose_landmarker_full",
    "heavy": "pose_landmarker_heavy",
}
BASE = "https://storage.googleapis.com/mediapipe-models/pose_landmarker"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def url_for(variant: str) -> str:
    name = MODELS[variant]
    return f"{BASE}/{name}/float16/latest/{name}.task"


def _progress(block_num: int, block_size: int, total: int) -> None:
    if total <= 0:
        return
    pct = min(100, block_num * block_size * 100 // total)
    print(f"\r  downloading... {pct:3d}%", end="", flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download a PoseLandmarker model.")
    p.add_argument("--variant", choices=MODELS, default="full")
    args = p.parse_args(argv)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / f"{MODELS[args.variant]}.task"
    if dest.exists():
        print(f"Already present: {dest}")
        return 0

    url = url_for(args.variant)
    print(f"Fetching {args.variant} model:\n  {url}")
    try:
        urllib.request.urlretrieve(url, dest, _progress)
    except Exception as e:  # noqa: BLE001 - surface any network/IO error
        print(f"\nerror: download failed: {e}", file=sys.stderr)
        if dest.exists():
            dest.unlink()
        return 1
    print(f"\nSaved to {dest} ({dest.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
