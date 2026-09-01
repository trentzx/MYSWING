"""Golf swing analyzer v1.

Pipeline: video -> MediaPipe pose keypoints -> swing-phase detection ->
metrics -> annotated output video.
"""
from .pose_extraction import extract_poses, PoseFrame, VideoMeta
from .phases import detect_phases, SwingPhases
from .metrics import compute_metrics, SwingMetrics
from .overlay import render_overlay

__all__ = [
    "extract_poses",
    "PoseFrame",
    "VideoMeta",
    "detect_phases",
    "SwingPhases",
    "compute_metrics",
    "SwingMetrics",
    "render_overlay",
]

__version__ = "1.0.0"
