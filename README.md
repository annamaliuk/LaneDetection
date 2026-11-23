# Lane Detection and Tracking

Detects and tracks the **left and right ego-lane lines** in video from a windshield camera.
Two neural detectors propose lane candidates, a learned rescoring model picks the reliable ones,
and a custom Kalman filter keeps the lines stable over time. The demo runs on a laptop CPU.

## How it works

1. **Preprocessing.** The top 40% of the frame (sky, buildings) is cropped and the rest is resized to 640×224.
2. **Row-anchor detector (ResNet-18, INT8).** For 36 fixed image rows the network predicts where the line is
   (120 horizontal cells) and whether it exists. The spread of the prediction gives an uncertainty σ per row.
3. **Second detector (YOLO26n-seg).** Segments each line as a mask; the mask is turned into a line with a
   RANSAC parabola fit.
4. **Rescoring.** Up to two candidates per side (one from each detector) are described by 15 features
   (confidence, geometry, σ, agreement between detectors, lane width). A gradient boosting model
   (HistGradientBoosting) predicts whether each candidate is correct and keeps the best one.
5. **Kalman filter tracker (NumPy).** State: x at three rows plus velocities. The detector's σ sets the
   measurement noise. Includes χ² gating, left/right swap fix, lane-width coupling, coasting through missed
   detections and lane-change handling.

## Results

Balanced CULane test set: 3,798 frames, 9 conditions × 422 frames. Metric: CULane F1 (line IoU > 0.5),
95% bootstrap confidence intervals over video clips.

| Pipeline | Test F1 | 95% CI | Precision |
| --- | --- | --- | --- |
| Classical baseline (bird's-eye view, Sobel, sliding windows) | 0.089 | | |
| ResNet-18 row-anchor | 0.656 | 0.625–0.683 | 0.643 |
| YOLO26n-seg | 0.663 | 0.631–0.690 | 0.778 |
| Ensemble (ResNet-18 + DINOv3 + ResNet-34, flip TTA) | 0.693 | 0.658–0.722 | 0.683 |
| ResNet-18 + YOLO + rescoring (**demo**) | 0.708 | 0.679–0.734 | 0.806 |
| Ensemble + YOLO + DINOv3 embedding + rescoring (**best**) | **0.731** | 0.702–0.756 | 0.803 |

- Rescoring cuts false lines at crossroads in half (from 273 to 135 on 422 frames without lines).
- Tracker on synthetic sequences with ground truth: F1 from 0.860 to 0.974, jitter from 10.4 to 1.8 px,
  left/right swaps from 12.7 to 0 per 1,000 frames. On real CULane video: jitter from 1.40 to 0.79 px.
- Speed (ONNX Runtime, CPU, 2 threads): ResNet-18 takes 121 ms in FP32 and 65 ms in INT8, for −0.5 F1.
  Measured time follows the theoretical complexity: t = 12.40 · GFLOPs − 0.72 ms (R² = 1.000).
- Cross-domain test on BDD100K: F1 ≈ 0.15 for all models (different cameras, roads and labelling style).

## Repository structure

```
demo/        demo application (Gradio) and stress test
models/      model files (not in the repository, see models/README.md)
notebooks/   training and evaluation notebooks (Google Colab)
results/     result tables (CSV) and figures
```

| Notebook | Purpose |
| --- | --- |
| `preprocessing_1.ipynb` | Cropping, resizing, ego-line selection, balanced splits |
| `metric_and_baseline.ipynb` | CULane metric, sanity checks, classical baseline |
| `resnet_lane.ipynb` | Row-anchor ResNet-18, resolution sweep |
| `dino_lane.ipynb` | DINOv3 fine-tuning, ensemble, distillation |
| `best_lane.ipynb` | Smoothing, ResNet-34, final ensemble |
| `yolo_lane.ipynb` | YOLO26n-seg training, mask-to-line conversion |
| `tracking.ipynb` | Kalman filter, synthetic tests, parameter search |
| `eval_lane.ipynb` | Test sets, per-condition results, INT8, CPU speed |
| `fusion_lane.ipynb` | Candidates, features, rescoring model |

## Running the demo

Tested on macOS (Intel) with **Python 3.12**. On Intel Macs PyTorch is only available up to 2.2.2,
which needs `numpy<2`.

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download the model files into `models/` (see `models/README.md`):
   `model_int8.onnx`, `best.pt`, `meta.joblib`, and optionally an example video `example.mp4`.
3. Start the app from the repository root:
   ```bash
   python demo/app.py
   ```
   The browser opens a page where you upload a video, switch the tracker on or off and get back the video
   with drawn lines (blue: left, orange: right, yellow: predicted by the tracker), a summary table and a
   per-frame CSV. Outputs are saved to `outputs/`.
4. Optional stress test on worst-case inputs (black frames, noise, night, tiny and 4K video, broken files):
   ```bash
   python demo/worst_case.py
   ```

If the example video does not play in the browser, convert it to H.264 with `python demo/convert_example.py`.

## Reproducing the training

Run the notebooks in Google Colab in the order of the table above. They expect CULane and BDD100K on
Google Drive under `lane detection/data/` and save models and results to `lane detection/models/` and
`lane detection/results/`. Training was done on an NVIDIA L4 GPU.

## Limitations

- Pixel accuracy is limited by the row-anchor grid (36 rows, 8 px cells): F1 at IoU 0.7 is about 0.59.
- Weak cross-domain generalisation (BDD100K).
- The offset in metres assumes a 3.7 m lane and an uncalibrated camera.
- About 6–8 FPS on an Intel MacBook CPU, below real time.

## License

YOLO26 comes from Ultralytics and is licensed under AGPL-3.0.
