"""Command-line entry point: video -> phases -> metrics -> annotated video."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import DEFAULT_MODEL_PATH
from .metrics import compute_metrics
from .phases import detect_phases
from .pose_extraction import extract_poses
from .overlay import render_overlay


def _default_output(input_video: Path) -> Path:
    return Path("data/output") / f"{input_video.stem}_analyzed.mp4"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="golf_analyzer",
        description="Golf swing analyzer v1 (down-the-line, MediaPipe Tasks API).",
    )
    p.add_argument("input", type=Path, help="Path to the swing video.")
    p.add_argument(
        "-m", "--model", type=Path, default=DEFAULT_MODEL_PATH,
        help=f"PoseLandmarker .task model (default: {DEFAULT_MODEL_PATH}).",
    )
    p.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Annotated output video path (default: data/output/<name>_analyzed.mp4).",
    )
    p.add_argument(
        "--json", type=Path, default=None,
        help="Optional path to write metrics + phases as JSON.",
    )
    p.add_argument(
        "--no-video", action="store_true",
        help="Skip rendering the annotated video (metrics only).",
    )
    return p


def run(args: argparse.Namespace) -> dict:
    print(f"[1/4] Extracting poses from {args.input} ...")
    frames, meta = extract_poses(args.input, args.model)
    detected = sum(1 for f in frames if f.present)
    print(
        f"      {meta.frame_count} frames @ {meta.fps:.1f} fps "
        f"({meta.width}x{meta.height}); pose found in {detected}/{meta.frame_count}."
    )

    print("[2/4] Detecting swing phases ...")
    phases = detect_phases(frames, meta)
    print(
        f"      address=f{phases.address}  top=f{phases.top}  impact=f{phases.impact}"
    )

    print("[3/4] Computing metrics ...")
    metrics = compute_metrics(frames, meta, phases)
    print("      " + "-" * 44)
    print(f"      Tempo (backswing:downswing) : {metrics.tempo_label} "
          f"({metrics.backswing_frames}:{metrics.downswing_frames} frames)")
    print(f"      Head stability              : {metrics.head_stability_pct:.1f}% "
          f"of shoulder width ({metrics.head_displacement_px:.0f}px)")
    print(f"      Spine tilt @ address        : {metrics.spine_tilt_deg:.1f} deg "
          f"from vertical")
    print("      " + "-" * 44)

    result = {
        "input": str(args.input),
        "video": {
            "fps": meta.fps,
            "width": meta.width,
            "height": meta.height,
            "frame_count": meta.frame_count,
        },
        "phases": {
            "address": phases.address,
            "top": phases.top,
            "impact": phases.impact,
        },
        "metrics": metrics.to_dict(),
    }

    if not args.no_video:
        out = args.output or _default_output(args.input)
        print(f"[4/4] Rendering annotated video -> {out} ...")
        render_overlay(args.input, out, frames, meta, phases, metrics)
        result["output_video"] = str(out)
        print("      done.")
    else:
        print("[4/4] Skipping video render (--no-video).")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2))
        print(f"      Metrics written to {args.json}")

    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
