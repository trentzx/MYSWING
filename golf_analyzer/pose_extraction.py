"""Video -> MediaPipe pose keypoints.

Uses the MediaPipe Tasks API (PoseLandmarker with a .task model) in VIDEO
running mode. The legacy ``mp.solutions.pose`` module is intentionally not used
because it is absent from mediapipe 0.10.33+.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import mediapipe as mp

from .config import NUM_LANDMARKS

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


@dataclass
class VideoMeta:
    fps: float
    width: int
    height: int
    frame_count: int


@dataclass
class PoseFrame:
    """Pose result for a single video frame.

    ``landmarks`` is an (33, 4) array of normalized image coordinates:
    columns are (x, y, z, visibility). x/y are in [0, 1] relative to frame
    width/height. ``present`` is False when no pose was detected this frame
    (landmarks are then all-NaN so downstream code can interpolate/skip).
    """

    index: int
    timestamp_ms: int
    landmarks: np.ndarray
    present: bool

    def xy_pixels(self, meta: VideoMeta) -> np.ndarray:
        """Return an (33, 2) array of pixel coordinates for this frame."""
        px = self.landmarks[:, :2].copy()
        px[:, 0] *= meta.width
        px[:, 1] *= meta.height
        return px


def _empty_landmarks() -> np.ndarray:
    arr = np.full((NUM_LANDMARKS, 4), np.nan, dtype=np.float32)
    return arr


def extract_poses(
    video_path: str | Path,
    model_path: str | Path,
    *,
    min_pose_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> tuple[list[PoseFrame], VideoMeta]:
    """Run PoseLandmarker over every frame of ``video_path``.

    Returns the per-frame pose results and the video metadata.
    """
    video_path = Path(video_path)
    model_path = Path(model_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Pose model not found: {model_path}\n"
            "Download it with: python scripts/download_model.py"
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        fps = 30.0  # sane fallback if the container lacks fps metadata
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    meta = VideoMeta(fps=fps, width=width, height=height, frame_count=frame_count)

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    frames: list[PoseFrame] = []
    with PoseLandmarker.create_from_options(options) as landmarker:
        idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            # Timestamps must be monotonically increasing integers (ms).
            timestamp_ms = int(round(idx * 1000.0 / fps))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                arr = np.array(
                    [[p.x, p.y, p.z, p.visibility] for p in lm],
                    dtype=np.float32,
                )
                frames.append(PoseFrame(idx, timestamp_ms, arr, present=True))
            else:
                frames.append(
                    PoseFrame(idx, timestamp_ms, _empty_landmarks(), present=False)
                )
            idx += 1

    cap.release()

    # Correct frame_count to what we actually read (container metadata lies).
    meta.frame_count = len(frames)
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return frames, meta


def landmark_series(frames: list[PoseFrame], index: int) -> np.ndarray:
    """Stack one landmark's (x, y, z, visibility) across all frames -> (N, 4)."""
    return np.array([f.landmarks[index] for f in frames], dtype=np.float32)
