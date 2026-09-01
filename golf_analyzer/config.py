"""Shared constants for the golf swing analyzer.

Landmark indices follow MediaPipe's 33-point BlazePose topology (same ordering
whether you use the legacy solution or the Tasks API PoseLandmarker).
"""
from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "pose_landmarker_full.task"

# --- Landmark indices ------------------------------------------------------
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24

NUM_LANDMARKS = 33

# Visibility below this is treated as "landmark not reliably seen this frame".
VISIBILITY_THRESHOLD = 0.5

# Pose skeleton connections (subset of BlazePose POSE_CONNECTIONS) used for the
# overlay. Kept local so the renderer doesn't depend on mp.solutions internals.
POSE_CONNECTIONS = [
    # face (minimal)
    (0, 2), (0, 5),
    # arms
    (11, 13), (13, 15), (12, 14), (14, 16),
    # shoulders / torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # legs
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
]
