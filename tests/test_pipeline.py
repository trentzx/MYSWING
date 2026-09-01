"""Logic tests using synthetic pose data (no MediaPipe / no real video needed).

These validate the fps-independent phase detection and the metric math against
a hand-built swing whose ground-truth phases we know.
"""
from __future__ import annotations

import numpy as np

from golf_analyzer.config import (
    NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
    LEFT_WRIST, RIGHT_WRIST, NUM_LANDMARKS,
)
from golf_analyzer.pose_extraction import PoseFrame, VideoMeta
from golf_analyzer.phases import detect_phases
from golf_analyzer.metrics import compute_metrics

W, H = 1280, 720
FPS = 120.0

# Ground-truth phase frames for the synthetic swing.
GT_ADDRESS_REGION = (10, 20)  # detected address should fall in the quiet run
GT_TOP = 55
GT_IMPACT = 68


def _synthetic_wrist_y(n: int) -> np.ndarray:
    """Wrist height (normalized y, 0=top of frame). Low->up->down->up again."""
    y = np.empty(n)
    y[:20] = 0.70                                   # address: low & still
    y[20:GT_TOP] = np.linspace(0.70, 0.30, GT_TOP - 20)   # backswing rising
    y[GT_TOP:GT_IMPACT] = np.linspace(0.30, 0.68, GT_IMPACT - GT_TOP)  # downswing
    y[GT_IMPACT:] = np.linspace(0.68, 0.25, n - GT_IMPACT)  # follow-through high
    return y


def _synthetic_wrist_x(n: int) -> np.ndarray:
    x = np.empty(n)
    x[:20] = 0.50
    x[20:GT_TOP] = np.linspace(0.50, 0.62, GT_TOP - 20)
    x[GT_TOP:GT_IMPACT] = np.linspace(0.62, 0.50, GT_IMPACT - GT_TOP)
    x[GT_IMPACT:] = np.linspace(0.50, 0.40, n - GT_IMPACT)
    return x


def _build_frames(n: int = 100) -> list[PoseFrame]:
    wy = _synthetic_wrist_y(n)
    wx = _synthetic_wrist_x(n)
    frames: list[PoseFrame] = []
    for i in range(n):
        lm = np.zeros((NUM_LANDMARKS, 4), dtype=np.float32)
        lm[:, 3] = 1.0  # full visibility everywhere

        lm[NOSE] = [0.50, 0.20, 0.0, 1.0]
        # Shoulders shifted +x of hips -> forward spine tilt (down-the-line).
        lm[LEFT_SHOULDER] = [0.50, 0.35, 0.0, 1.0]
        lm[RIGHT_SHOULDER] = [0.60, 0.35, 0.0, 1.0]
        lm[LEFT_HIP] = [0.45, 0.60, 0.0, 1.0]
        lm[RIGHT_HIP] = [0.55, 0.60, 0.0, 1.0]
        # Both wrists at the same (midpoint) location for simplicity.
        lm[LEFT_WRIST] = [wx[i], wy[i], 0.0, 1.0]
        lm[RIGHT_WRIST] = [wx[i], wy[i], 0.0, 1.0]

        ts = int(round(i * 1000.0 / FPS))
        frames.append(PoseFrame(i, ts, lm, present=True))
    return frames


def test_phase_detection_picks_correct_events():
    frames = _build_frames()
    meta = VideoMeta(fps=FPS, width=W, height=H, frame_count=len(frames))
    phases = detect_phases(frames, meta)

    assert GT_ADDRESS_REGION[0] <= phases.address <= GT_ADDRESS_REGION[1], phases.address
    # Top (min y) should be near GT_TOP and NOT the higher follow-through.
    assert abs(phases.top - GT_TOP) <= 3, phases.top
    assert phases.top < phases.impact
    # Impact is the downswing speed peak, before the follow-through.
    assert GT_TOP < phases.impact < 75, phases.impact


def test_metrics_are_sane():
    frames = _build_frames()
    meta = VideoMeta(fps=FPS, width=W, height=H, frame_count=len(frames))
    phases = detect_phases(frames, meta)
    m = compute_metrics(frames, meta, phases)

    # Tempo: backswing should be longer than downswing here -> ratio > 1.
    assert m.backswing_frames == phases.top - phases.address
    assert m.downswing_frames == phases.impact - phases.top
    assert m.tempo_ratio > 1.0

    # Shoulder width = 0.10 * 1280 = 128 px.
    assert abs(m.shoulder_width_px - 128.0) < 1.0

    # Nose is fixed in this synthetic swing -> stability ~0.
    assert m.head_stability < 0.01

    # Spine tilt: shoulder_mid x=0.55, hip_mid x=0.50 -> dx=64px, dy=180px.
    expected = np.degrees(np.arctan2(64.0, 180.0))  # ~19.6 deg
    assert abs(m.spine_tilt_deg - expected) < 1.0


if __name__ == "__main__":
    test_phase_detection_picks_correct_events()
    test_metrics_are_sane()
    print("All synthetic-data tests passed.")
