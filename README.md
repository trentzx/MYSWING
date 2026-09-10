# Golf Swing Analyzer (v1)

Analyzes a **down-the-line** golf swing video and reports three metrics, then
renders an annotated output video.

**Pipeline:** `video → MediaPipe pose keypoints → swing-phase detection → metrics → annotated video`

Uses MediaPipe's **Tasks API** (`PoseLandmarker` with a `.task` model file), not
the legacy `mp.solutions.pose` (absent from mediapipe 0.10.33+).

## Web frontend

The responsive **MYSWING Swing Studio** lives in `frontend/`. It uses plain
HTML, CSS, and JavaScript, with no npm dependencies or build step.

```powershell
python -m http.server 5173 --bind 127.0.0.1 --directory frontend
```

Open http://localhost:5173. Choose or drag in an MP4, MOV, or WebM video
(up to 250 MB) to review it locally. Codec support depends on your browser;
H.264 MP4 is recommended. Videos and reports are not uploaded or saved.

To see measured results, generate a report with the existing analyzer:

```powershell
python -m golf_analyzer data/input/my_swing.mp4 --json data/output/my_swing.json
```

Use **Import analysis report** to load the JSON file. The studio displays
tempo, head stability, spine tilt, and recording-confidence warnings. Load
the original or annotated video with the filename recorded in the report
to enable Address, Top, and Impact navigation. Keep matching videos and
reports together; filename matching does not verify video contents.

The frontend is a review interface: selecting a video does **not** run
MediaPipe in the browser. Analysis still runs through the Python CLI.
You can host the contents of `frontend/` on any static website host.
Google Fonts supplies the typefaces, with local sans-serif fallbacks.

Frontend syntax check: `node --check frontend/app.js`.

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

## Input handling & reliability

v1 targets a **single, continuous down-the-line swing**. Before analyzing, the
tool segments the clip to find the actual swing and guards against unsuitable
input:

* Detects hard **scene cuts** (frame differencing) and splits the clip into shots.
* Picks the shot containing the swing (strongest hand-speed burst with the body in frame).
* Emits **warnings** for low frame rate, poor pose coverage, or edited/multi-shot
  (broadcast) footage.
* Sets a **`reliable` flag** — when the input isn't a clean single swing (e.g. a
  TV analysis segment with replays and camera cuts), metrics are still reported
  but clearly marked **LOW CONFIDENCE**.

> Note: cleanly extracting one swing from a long broadcast montage is a harder,
> later problem. For trustworthy numbers, feed one uncut swing.

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
  segmentation.py     locate the swing in a clip + input-quality guardrails
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
