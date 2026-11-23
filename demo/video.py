import time
import numpy as np, pandas as pd, cv2, imageio
from config import OUT_DIR, W, H, CROP
from detector import detect
from tracker import Kalman

COLORS = [(255, 120, 0), (0, 120, 255)]


def to_frame(lane, w, h, crop):
	rows = np.where(np.isfinite(lane))[0]
	return np.stack([np.clip(lane[rows] * w / W, -5 * w, 6 * w), rows * (h - crop) / H + crop], 1).astype(np.int32)


def offset_m(lanes):
	both = np.where(np.isfinite(lanes[0]) & np.isfinite(lanes[1]))[0]
	if len(both) == 0 or lanes[1, both[-1]] - lanes[0, both[-1]] < 20:
		return np.nan
	left, right = lanes[0, both[-1]], lanes[1, both[-1]]
	return (W / 2 - (left + right) / 2) * 3.7 / (right - left)


def draw(frame, lanes, coasting, crop, text):
	h, w = frame.shape[:2]
	both = np.isfinite(lanes[0]) & np.isfinite(lanes[1])
	if both.sum() > 1:
		area = np.concatenate([to_frame(np.where(both, lanes[0], np.nan), w, h, crop), to_frame(np.where(both, lanes[1], np.nan), w, h, crop)[::-1]])
		overlay = cv2.fillPoly(frame.copy(), [area], (0, 200, 0))
		frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
	for k in range(2):
		cv2.polylines(frame, [to_frame(lanes[k], w, h, crop)], False, (0, 220, 255) if coasting[k] else COLORS[k], 3)
	for j, line in enumerate(text):
		cv2.putText(frame, line, (10, 30 + 30 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
	return frame


def process_video(path, use_tracker=True, max_seconds=20):
	cap = cv2.VideoCapture(str(path))
	fps = cap.get(cv2.CAP_PROP_FPS)
	fps = fps if 1 <= fps <= 240 else 25.0
	tracker = Kalman() if use_tracker else None
	name = f"{OUT_DIR}/lanes_{time.time_ns()}"
	writer, rows = None, []
	while len(rows) < max_seconds * fps:
		ok, frame = cap.read()
		if not ok:
			break
		h, w = frame.shape[:2]
		s = 1280 / w if w > 1280 else max(1.0, 320 / w)
		frame = cv2.resize(frame, (int(w * s) // 2 * 2, max(2, int(h * s) // 2 * 2)))
		crop = int(CROP * frame.shape[0])
		start = time.perf_counter()
		lanes, conf, sigma = detect(cv2.cvtColor(cv2.resize(frame[crop:], (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
		coasting = [False, False]
		if tracker:
			lanes, coasting = tracker.step(lanes, conf, sigma)
		ms = (time.perf_counter() - start) * 1000
		off = offset_m(lanes)
		frame = draw(frame, lanes, coasting, crop, [f"{1000 / ms:.1f} FPS", f"offset {off:+.2f} m"])
		writer = writer or imageio.get_writer(f"{name}.mp4", fps=fps, codec='libx264', macro_block_size=1)
		writer.append_data(frame[:, :, ::-1])
		rows.append({'both_lines': bool(np.isfinite(lanes).any(1).all()), 'coasting': any(coasting), 'offset_m': off, 'ms': ms})
	cap.release()
	if writer is None:
		raise ValueError('No readable frames in this file.')
	writer.close()
	df = pd.DataFrame(rows).round(3)
	df.to_csv(f"{name}.csv", index_label='frame')
	summary = pd.DataFrame([{'frames': len(df), 'ms per frame': round(df.ms.mean(), 1), 'FPS': round(1000 / df.ms.mean(), 1),
	                         'both lines %': round(100 * df.both_lines.mean(), 1), 'coasted %': round(100 * df.coasting.mean(), 1)}])
	return f"{name}.mp4", summary, f"{name}.csv"