"""Swing-phase detection from the wrist (hand) trajectory.

Heuristic (down-the-line view):
  * Track the wrist midpoint (mean of left/right wrist) in pixels.
  * impact  = frame of maximum hand speed. The downswing is by far the fastest
              motion in a golf swing, so its speed peak is a robust anchor.
  * top     = frame of minimum wrist height (min y) *before* impact. Anchoring
              to "before impact" avoids mistaking the high hands of the
              follow-through for the top of the backswing.
  * address = last quiet frame before the takeaway begins (scanning back from
              the top until hand speed drops near zero).

All frame indices, so tempo derived from them is fps-independent.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import LEFT_WRIST, RIGHT_WRIST, VISIBILITY_THRESHOLD
from .pose_extraction import PoseFrame, VideoMeta, landmark_series


@dataclass
class SwingPhases:
    address: int
    top: int
    impact: int
    # Diagnostics / overlay aids:
    wrist_xy: np.ndarray  # (N, 2) smoothed wrist-midpoint pixel track
    speed: np.ndarray     # (N,) smoothed hand speed (px/frame)


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """NaN-aware centered moving average that preserves array length."""
    if window <= 1:
        return x
    n = len(x)
    out = np.empty_like(x)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        seg = x[lo:hi]
        seg = seg[~np.isnan(seg)]
        out[i] = np.mean(seg) if seg.size else np.nan
    return out


def _interp_nans(x: np.ndarray) -> np.ndarray:
    """Linearly interpolate over NaNs so tracking survives dropped frames."""
    x = x.copy()
    nans = np.isnan(x)
    if nans.all():
        return x
    idx = np.arange(len(x))
    x[nans] = np.interp(idx[nans], idx[~nans], x[~nans])
    return x


def _wrist_midpoint_pixels(frames: list[PoseFrame], meta: VideoMeta) -> np.ndarray:
    """(N, 2) wrist-midpoint track in pixels, gated by visibility, gap-filled."""
    lw = landmark_series(frames, LEFT_WRIST)
    rw = landmark_series(frames, RIGHT_WRIST)

    # Zero-out low-visibility samples to NaN so they get interpolated.
    lw_vis = lw[:, 3] >= VISIBILITY_THRESHOLD
    rw_vis = rw[:, 3] >= VISIBILITY_THRESHOLD

    lx = np.where(lw_vis, lw[:, 0], np.nan)
    ly = np.where(lw_vis, lw[:, 1], np.nan)
    rx = np.where(rw_vis, rw[:, 0], np.nan)
    ry = np.where(rw_vis, rw[:, 1], np.nan)

    # Average whichever wrists are visible; if both NaN the mean stays NaN.
    mx = np.nanmean(np.vstack([lx, rx]), axis=0)
    my = np.nanmean(np.vstack([ly, ry]), axis=0)

    mx = _interp_nans(mx) * meta.width
    my = _interp_nans(my) * meta.height
    return np.vstack([mx, my]).T


def detect_phases(
    frames: list[PoseFrame],
    meta: VideoMeta,
    *,
    smooth_window: int = 5,
) -> SwingPhases:
    n = len(frames)
    if n < 5:
        raise ValueError("Video too short for phase detection (need >= 5 frames).")

    wrist = _wrist_midpoint_pixels(frames, meta)
    wx = _moving_average(wrist[:, 0], smooth_window)
    wy = _moving_average(wrist[:, 1], smooth_window)
    wrist_s = np.vstack([wx, wy]).T

    # Frame-to-frame speed (px/frame); prepend 0 to keep length N.
    vel = np.diff(wrist_s, axis=0)
    speed = np.concatenate([[0.0], np.linalg.norm(vel, axis=1)])
    speed = _moving_average(speed, smooth_window)

    # --- impact: global speed peak -----------------------------------------
    impact = int(np.nanargmax(speed))

    # --- top: highest hands (min y) strictly before impact -----------------
    # Guard against impact landing at frame 0 (degenerate clips).
    search_end = max(impact, 1)
    top = int(np.nanargmin(wy[:search_end]))

    # --- address: last quiet frame before the takeaway ---------------------
    peak_speed = float(np.nanmax(speed))
    quiet_thresh = max(0.5, 0.06 * peak_speed)  # px/frame
    address = 0
    for i in range(top - 1, -1, -1):
        if speed[i] < quiet_thresh:
            address = i
            break

    # Sanity: enforce address < top < impact.
    if not (address < top < impact):
        address = min(address, max(0, top - 1))
        top = min(max(top, address + 1), impact - 1)

    return SwingPhases(
        address=address,
        top=top,
        impact=impact,
        wrist_xy=wrist_s,
        speed=speed,
    )
