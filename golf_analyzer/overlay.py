"""Annotated output video: skeleton, phase markers, wrist trail, metrics panel."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import POSE_CONNECTIONS, VISIBILITY_THRESHOLD
from .metrics import SwingMetrics
from .phases import SwingPhases
from .pose_extraction import PoseFrame, VideoMeta

# BGR colors
_SKELETON = (0, 255, 0)
_JOINT = (0, 200, 255)
_TRAIL = (255, 180, 0)
_PANEL_BG = (0, 0, 0)
_TEXT = (255, 255, 255)
_PHASE_COLORS = {
    "ADDRESS": (0, 255, 255),
    "TOP": (0, 165, 255),
    "IMPACT": (0, 0, 255),
}


def _draw_skeleton(img: np.ndarray, frame: PoseFrame, meta: VideoMeta) -> None:
    if not frame.present:
        return
    px = frame.xy_pixels(meta)
    vis = frame.landmarks[:, 3]
    for a, b in POSE_CONNECTIONS:
        if vis[a] < VISIBILITY_THRESHOLD or vis[b] < VISIBILITY_THRESHOLD:
            continue
        pa = tuple(np.round(px[a]).astype(int))
        pb = tuple(np.round(px[b]).astype(int))
        cv2.line(img, pa, pb, _SKELETON, 2, cv2.LINE_AA)
    for i in range(len(px)):
        if vis[i] < VISIBILITY_THRESHOLD:
            continue
        p = tuple(np.round(px[i]).astype(int))
        cv2.circle(img, p, 3, _JOINT, -1, cv2.LINE_AA)


def _draw_wrist_trail(
    img: np.ndarray, wrist_xy: np.ndarray, upto: int, tail: int = 45
) -> None:
    start = max(0, upto - tail)
    pts = wrist_xy[start : upto + 1]
    for i in range(1, len(pts)):
        if np.any(np.isnan(pts[i])) or np.any(np.isnan(pts[i - 1])):
            continue
        alpha = i / len(pts)
        color = tuple(int(c * alpha) for c in _TRAIL)
        cv2.line(
            img,
            tuple(np.round(pts[i - 1]).astype(int)),
            tuple(np.round(pts[i]).astype(int)),
            color,
            2,
            cv2.LINE_AA,
        )


def _put_lines(img, lines, org, scale=0.6, color=_TEXT, thick=1, dy=24) -> None:
    x, y = org
    for i, line in enumerate(lines):
        cv2.putText(
            img,
            line,
            (x, y + i * dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thick,
            cv2.LINE_AA,
        )


def _draw_panel(img: np.ndarray, metrics: SwingMetrics) -> None:
    h, w = img.shape[:2]
    pw = min(360, int(w * 0.42))
    ph = 150
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (pw, ph), _PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    lines = [
        f"Tempo (back:down): {metrics.tempo_label}",
        f"   {metrics.backswing_frames} : {metrics.downswing_frames} frames",
        f"Head stability: {metrics.head_stability_pct:.1f}% of shoulders",
        f"Spine tilt @ address: {metrics.spine_tilt_deg:.1f} deg",
    ]
    _put_lines(img, lines, (12, 28), scale=0.55, dy=26)


def _phase_label_for(index: int, phases: SwingPhases) -> str | None:
    if index == phases.address:
        return "ADDRESS"
    if index == phases.top:
        return "TOP"
    if index == phases.impact:
        return "IMPACT"
    return None


def render_overlay(
    input_video: str | Path,
    output_video: str | Path,
    frames: list[PoseFrame],
    meta: VideoMeta,
    phases: SwingPhases,
    metrics: SwingMetrics,
    *,
    hold_phase_frames: int = 8,
) -> Path:
    """Re-read the source video and write an annotated copy.

    Phase labels linger for ``hold_phase_frames`` after each key event so they
    are readable even in fast slow-mo footage.
    """
    input_video = Path(input_video)
    output_video = Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not reopen video: {input_video}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_video), fourcc, meta.fps, (meta.width, meta.height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter for {output_video}")

    phase_events = [
        (phases.address, "ADDRESS"),
        (phases.top, "TOP"),
        (phases.impact, "IMPACT"),
    ]

    idx = 0
    while True:
        ok, img = cap.read()
        if not ok or idx >= len(frames):
            break

        _draw_wrist_trail(img, phases.wrist_xy, idx)
        _draw_skeleton(img, frames[idx], meta)

        # Persistent, lingering phase banner.
        active = None
        for fidx, name in phase_events:
            if fidx <= idx < fidx + hold_phase_frames:
                active = name
        if active:
            color = _PHASE_COLORS[active]
            cv2.rectangle(img, (0, 0), (meta.width, meta.height), color, 6)
            (tw, th), _ = cv2.getTextSize(
                active, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3
            )
            cx = (meta.width - tw) // 2
            cv2.putText(
                img, active, (cx, meta.height - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA,
            )

        _draw_panel(img, metrics)
        writer.write(img)
        idx += 1

    cap.release()
    writer.release()
    return output_video
