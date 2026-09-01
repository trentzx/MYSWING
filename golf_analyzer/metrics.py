"""v1 swing metrics: tempo ratio, head stability, spine tilt at address."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from .config import (
    NOSE,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_HIP,
    RIGHT_HIP,
)
from .phases import SwingPhases
from .pose_extraction import PoseFrame, VideoMeta


@dataclass
class SwingMetrics:
    # Tempo
    backswing_frames: int
    downswing_frames: int
    tempo_ratio: float          # backswing / downswing
    tempo_label: str            # e.g. "3.0 : 1"
    # Head stability
    head_displacement_px: float
    shoulder_width_px: float
    head_stability: float       # displacement / shoulder width (lower = steadier)
    head_stability_pct: float   # same, as a percentage
    # Spine tilt
    spine_tilt_deg: float       # degrees from vertical at address

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _midpoint_px(frame: PoseFrame, meta: VideoMeta, a: int, b: int) -> np.ndarray:
    px = frame.xy_pixels(meta)
    return (px[a] + px[b]) / 2.0


def _tempo(phases: SwingPhases) -> tuple[int, int, float, str]:
    backswing = phases.top - phases.address
    downswing = phases.impact - phases.top
    ratio = backswing / downswing if downswing > 0 else float("nan")
    label = f"{ratio:.1f} : 1" if np.isfinite(ratio) else "n/a"
    return backswing, downswing, ratio, label


def _head_stability(
    frames: list[PoseFrame],
    meta: VideoMeta,
    phases: SwingPhases,
) -> tuple[float, float, float, float]:
    # Shoulder width in pixels measured at address (a stable, square stance).
    addr = frames[phases.address]
    ls = addr.xy_pixels(meta)[LEFT_SHOULDER]
    rs = addr.xy_pixels(meta)[RIGHT_SHOULDER]
    shoulder_w = float(np.linalg.norm(ls - rs))

    # Nose displacement across the active swing window (address -> impact),
    # measured as the max distance from the address-frame nose position.
    nose_addr = addr.xy_pixels(meta)[NOSE]
    max_disp = 0.0
    for f in frames[phases.address : phases.impact + 1]:
        nose = f.xy_pixels(meta)[NOSE]
        if np.any(np.isnan(nose)):
            continue
        d = float(np.linalg.norm(nose - nose_addr))
        max_disp = max(max_disp, d)

    stability = max_disp / shoulder_w if shoulder_w > 0 else float("nan")
    return max_disp, shoulder_w, stability, stability * 100.0


def _spine_tilt_at_address(
    frames: list[PoseFrame], meta: VideoMeta, phases: SwingPhases
) -> float:
    """Angle of the hip-midpoint -> shoulder-midpoint vector from vertical."""
    addr = frames[phases.address]
    shoulder_mid = _midpoint_px(addr, meta, LEFT_SHOULDER, RIGHT_SHOULDER)
    hip_mid = _midpoint_px(addr, meta, LEFT_HIP, RIGHT_HIP)

    # Vector pointing up the spine (hips -> shoulders). Image y grows downward,
    # so a straight-up spine is (0, -1).
    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]

    # Angle between spine vector and the vertical axis, in degrees.
    angle = np.degrees(np.arctan2(abs(dx), abs(dy)))
    return float(angle)


def compute_metrics(
    frames: list[PoseFrame],
    meta: VideoMeta,
    phases: SwingPhases,
) -> SwingMetrics:
    backswing, downswing, ratio, label = _tempo(phases)
    disp, shoulder_w, stability, stability_pct = _head_stability(frames, meta, phases)
    spine = _spine_tilt_at_address(frames, meta, phases)

    return SwingMetrics(
        backswing_frames=backswing,
        downswing_frames=downswing,
        tempo_ratio=round(ratio, 3),
        tempo_label=label,
        head_displacement_px=round(disp, 2),
        shoulder_width_px=round(shoulder_w, 2),
        head_stability=round(stability, 4),
        head_stability_pct=round(stability_pct, 2),
        spine_tilt_deg=round(spine, 2),
    )
