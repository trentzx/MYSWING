# Golf Swing Analyzer (v1)

Analyzes a **down-the-line** golf swing video and reports three metrics, then
renders an annotated output video.

**Pipeline:** `video → MediaPipe pose keypoints → swing-phase detection → metrics → annotated video`

Uses MediaPipe's **Tasks API** (`PoseLandmarker` with a `.task` model file), not
the legacy `mp.solutions.pose` (absent from mediapipe 0.10.33+).

## Metrics (v1)

| Metric | Definition |
| --- | --- |
| **Tempo ratio** | Backswing frames : downswing frames (`address→top` : `top→impact`). Frame-count based, so it's fps-independent. |
| **Head stability** | Max nose displacement during the swing, normalized by shoulder width (reported as % of shoulder width; lower = steadier). |
| **Spine tilt @ address** | Angle of the hip-midpoint→shoulder-midpoint vector from vertical, in degrees. |

## Target input

Down-the-line camera angle, 60fps+ slow-motion, plain background, full body in frame.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\download_model.py            # downloads models\pose_landmarker_full.task
```

`download_model.py` also accepts `--variant lite` (faster) or `--variant heavy`
(most accurate).

## Usage

```powershell
# Full run: metrics + annotated video (-> data\output\<name>_analyzed.mp4)
python -m golf_analyzer data\input\my_swing.mp4

# Metrics only, also dump JSON
python -m golf_analyzer data\input\my_swing.mp4 --no-video --json data\output\my_swing.json

# Custom model / output
python -m golf_analyzer data\input\my_swing.mp4 -m models\pose_landmarker_heavy.task -o out.mp4
```

## Phase-detection heuristic

* Track the **wrist midpoint** (mean of both wrists) in pixels, gap-filled and smoothed.
* **impact** = frame of maximum hand speed (the downswing is the fastest motion).
* **top of backswing** = minimum wrist height (`min y`) *before* impact — anchoring
  to "before impact" avoids mistaking the high hands of the follow-through for the top.
* **address** = last quiet frame before the takeaway (scanning back from the top
  until hand speed drops near zero).

## Project layout

```
golf_analyzer/
  config.py           landmark indices, model path, skeleton connections
  pose_extraction.py  video -> per-frame pose (Tasks API, VIDEO mode)
  phases.py           wrist trajectory -> address / top / impact
  metrics.py          tempo, head stability, spine tilt
  overlay.py          annotated video renderer
  cli.py              command-line entry point
scripts/download_model.py
tests/test_pipeline.py   synthetic-data logic tests (no video needed)
```

## Tests

```powershell
pytest                       # or: python tests\test_pipeline.py
```

The tests fabricate a synthetic swing with known ground-truth phases, so they
validate the detection/metric logic without needing a real video or GPU.
