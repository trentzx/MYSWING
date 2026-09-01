"""Locate the actual swing inside a clip and flag unsuitable input.

v1 assumes a single, continuous down-the-line swing filling the frame. Real
uploads (and broadcast clips like a TV "swing analysis" segment) contain
intros, multiple camera angles, replays and zooms. Running phase detection over
that whole timeline produces confident nonsense.

This module:
  * detects hard scene cuts via frame differencing,
  * splits the clip into shots and picks the one that actually contains the
    swing (highest hand-speed burst with a plausible up-then-fast-down shape),
  * assesses input quality and returns human-readable warnings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import LEFT_HIP, RIGHT_HIP, VISIBILITY_THRESHOLD
from .phases import wrist_track
from .pose_extraction import PoseFrame, VideoMeta

# A usable swing needs at least this many frames to show address->top->impact.
MIN_SWING_FRAMES = 12


@dataclass
class SwingLocation:
    window: tuple[int, int]       # [start, end) frame range containing the swing
    cuts: list[int]              # detected scene-cut frame indices
    num_shots: int
    reliable: bool = True        # False => metrics are low-confidence, don't trust
    warnings: list[str] = field(default_factory=list)


def frame_diff_series(
    video_path: str | Path, meta: VideoMeta, *, small: int = 128
) -> np.ndarray:
    """Mean absolute difference between consecutive (downscaled, gray) frames.

    Length N, first element 0. Hard cuts show up as sharp spikes.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for cut detection: {video_path}")

    scale_h = small
    diffs = [0.0]
    prev = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = g.shape
        scale_w = max(1, int(w * scale_h / max(1, h)))
        g = cv2.resize(g, (scale_w, scale_h))
        if prev is not None:
            diffs.append(float(np.mean(np.abs(g.astype(np.int16) - prev))))
        prev = g.astype(np.int16)
    cap.release()

    # Guard length against off-by-one vs the pose frame count.
    arr = np.array(diffs, dtype=np.float32)
    if len(arr) < meta.frame_count:
        arr = np.pad(arr, (0, meta.frame_count - len(arr)))
    return arr[: meta.frame_count]


def detect_cuts(diffs: np.ndarray, *, abs_floor: float = 18.0, k: float = 8.0) -> list[int]:
    """Frame indices where a hard scene cut occurs.

    A cut is a diff that is both large in absolute terms and a strong outlier
    versus the clip's typical frame-to-frame motion (median + k*MAD).
    """
    if len(diffs) < 3:
        return []
    body = diffs[1:]
    median = float(np.median(body))
    mad = float(np.median(np.abs(body - median))) or 1.0
    thresh = max(abs_floor, median + k * mad)
    return [int(i) for i in range(1, len(diffs)) if diffs[i] > thresh]


def find_shots(n: int, cuts: list[int]) -> list[tuple[int, int]]:
    """Split ``[0, n)`` into contiguous shots at the given cut indices."""
    bounds = [0] + sorted(c for c in cuts if 0 < c < n) + [n]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def _hips_visible_ratio(frames: list[PoseFrame], lo: int, hi: int) -> float:
    seg = frames[lo:hi]
    if not seg:
        return 0.0
    vis = [
        1.0
        for f in seg
        if f.present
        and f.landmarks[LEFT_HIP, 3] >= VISIBILITY_THRESHOLD
        and f.landmarks[RIGHT_HIP, 3] >= VISIBILITY_THRESHOLD
    ]
    return len(vis) / len(seg)


def _shot_swing_score(
    frames: list[PoseFrame], meta: VideoMeta, shot: tuple[int, int], speed: np.ndarray
) -> float:
    """How swing-like a shot is: peak hand speed, gated by length & body view."""
    lo, hi = shot
    if hi - lo < MIN_SWING_FRAMES:
        return 0.0
    if _hips_visible_ratio(frames, lo, hi) < 0.5:
        return 0.0
    return float(np.nanmax(speed[lo:hi]))


def choose_swing_shot(
    frames: list[PoseFrame], meta: VideoMeta, shots: list[tuple[int, int]]
) -> tuple[int, int]:
    """Pick the shot most likely to contain the swing (highest gated speed)."""
    _, speed = wrist_track(frames, meta)
    scored = [(_shot_swing_score(frames, meta, s, speed), s) for s in shots]
    scored.sort(key=lambda t: t[0], reverse=True)
    best_score, best = scored[0]
    if best_score <= 0.0:
        # Nothing looks swing-like; fall back to the longest shot.
        return max(shots, key=lambda s: s[1] - s[0])
    return best


def assess_quality(
    frames: list[PoseFrame],
    meta: VideoMeta,
    cuts: list[int],
    shots: list[tuple[int, int]],
    window: tuple[int, int],
) -> list[str]:
    warnings: list[str] = []

    if meta.fps < 60:
        warnings.append(
            f"Low frame rate ({meta.fps:.0f} fps). 60fps+ slow-motion is "
            "recommended for reliable phase timing."
        )

    detected = sum(1 for f in frames if f.present)
    if detected / max(1, len(frames)) < 0.9:
        warnings.append(
            f"Pose detected in only {detected}/{len(frames)} frames; the golfer "
            "may be partially out of frame or the background too busy."
        )

    if len(shots) > 1:
        warnings.append(
            f"{len(shots)} shots detected ({len(cuts)} scene cut(s)). This looks "
            "like edited/broadcast footage, not a single clean swing. Analyzing "
            f"only the swing segment (frames {window[0]}-{window[1]}). For best "
            "results upload one continuous, uncut swing."
        )

    lo, hi = window
    if hi - lo < MIN_SWING_FRAMES:
        warnings.append(
            "The located swing segment is very short; results may be unreliable."
        )

    if _hips_visible_ratio(frames, lo, hi) < 0.7:
        warnings.append(
            "The golfer's lower body isn't reliably visible in the swing "
            "segment; keep the full body in frame."
        )

    return warnings


def locate_swing(
    video_path: str | Path, frames: list[PoseFrame], meta: VideoMeta
) -> SwingLocation:
    """Full segmentation pass: cuts -> shots -> swing window -> warnings."""
    diffs = frame_diff_series(video_path, meta)
    cuts = detect_cuts(diffs)
    shots = find_shots(len(frames), cuts)
    window = choose_swing_shot(frames, meta, shots)
    warnings = assess_quality(frames, meta, cuts, shots, window)

    # Metrics are only trustworthy on a single, cleanly-detected swing with the
    # body in frame. Edited/multi-shot footage or poor pose coverage => the
    # chosen window likely spans multiple swings/replays, so flag low-confidence.
    lo, hi = window
    detected_ratio = sum(1 for f in frames if f.present) / max(1, len(frames))
    reliable = (
        len(shots) == 1
        and detected_ratio >= 0.85
        and _hips_visible_ratio(frames, lo, hi) >= 0.7
        and (hi - lo) >= MIN_SWING_FRAMES
    )

    return SwingLocation(
        window=window,
        cuts=cuts,
        num_shots=len(shots),
        reliable=reliable,
        warnings=warnings,
    )
