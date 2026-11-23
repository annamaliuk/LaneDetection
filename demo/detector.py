import numpy as np, pandas as pd, cv2, joblib
import onnxruntime as ort
from ultralytics import YOLO
from config import DEMO_DIR, W, H, ROWS
import compat

resnet = ort.InferenceSession(f"{DEMO_DIR}/model_int8.onnx", providers=['CPUExecutionProvider'])
yolo = YOLO(f"{DEMO_DIR}/best.pt")
fusion = joblib.load(f"{DEMO_DIR}/meta.joblib")
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def empty():
	return np.full((2, H), np.nan, np.float32), np.zeros(2, np.float32), np.full((2, H), np.nan, np.float32)


def fit(rows, x, deg, w=None, span=None):
	span = rows if span is None else span
	coef = np.polyfit(rows, x, deg, w=w)
	lane = np.full(H, np.nan, np.float32)
	r = np.arange(span.min(), span.max() + 1)
	lane[r] = np.polyval(coef, r)
	return lane, coef


def ransac(y, x):
	rng = np.random.default_rng(42)
	best = np.zeros(len(y), bool)
	for _ in range(60):
		s = rng.choice(len(y), 3, replace=False)
		inliers = np.abs(np.polyval(np.polyfit(y[s], x[s], 2), y) - x) < 3
		if inliers.sum() > best.sum():
			best = inliers
	return best


def run_resnet(img):
	x = ((img / 255 - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
	position, exist = resnet.run(None, {resnet.get_inputs()[0].name: x})[:2]
	p = np.exp(position[0] - position[0].max(-1, keepdims=True))
	p = p / p.sum(-1, keepdims=True)
	cells = np.arange(120)
	mean = (p * cells).sum(-1)
	spread = np.sqrt((p * (cells - mean[..., None]) ** 2).sum(-1)) * 8
	prob = 1 / (1 + np.exp(-exist[0]))
	lanes, conf, sigma = empty()
	for k in range(2):
		on = prob[k] > fusion['base_threshold']
		if on.sum() >= 4:
			r = np.arange(ROWS[on].min(), ROWS[on].max() + 1)
			sigma[k, r] = np.interp(r, ROWS[on], spread[k, on])
			lanes[k] = fit(ROWS[on], mean[k, on] * 8 - 156, 3, 1 / (spread[k, on] + 1))[0]
			conf[k] = prob[k, on].mean()
	return lanes, conf, sigma, prob


def run_yolo(img):
	lanes, conf, sigma = empty()
	r = yolo.predict(img[:, :, ::-1].copy(), imgsz=[H, W], conf=0.05, device='cpu', verbose=False)[0]
	if r.masks is None:
		return lanes, conf, sigma
	classes, scores = r.boxes.cls.numpy(), r.boxes.conf.numpy()
	for k in range(2):
		idx = np.where(classes == k)[0]
		if len(idx) == 0:
			continue
		best = idx[scores[idx].argmax()]
		polygon = r.masks.xy[best]
		if len(polygon) < 3:
			continue
		mask = np.zeros((H, W), np.uint8)
		cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
		rows, cols = np.nonzero(mask)
		valid = np.unique(rows)
		if len(rows) < 20 or len(valid) < 5:
			continue
		centers = np.bincount(rows, weights=cols)[valid] / np.bincount(rows)[valid]
		inliers = ransac(valid, centers)
		if inliers.sum() < 5:
			continue
		lanes[k], coef = fit(valid[inliers], centers[inliers], 2, span=valid)
		conf[k] = scores[best]
		sigma[k] = np.std(np.polyval(coef, valid) - centers)
	return lanes, conf, sigma


def agreement(a, b):
	both = np.isfinite(a) & np.isfinite(b)
	return np.abs(a[both] - b[both]).mean() if both.sum() >= 5 else np.nan


def detect(img):
	base = run_resnet(img)
	prob = base[3]
	sources = [base[:3], run_yolo(img)]
	features, cands = [], []
	for k in range(2):
		for s in range(2):
			lanes, conf, sigma = sources[s]
			r = np.where(np.isfinite(lanes[k]))[0]
			if len(r) < 2:
				continue
			other = sources[1 - s][0][k]
			features.append({'source': s, 'side': k, 'conf': conf[k], 'n_rows': len(r), 'top': r.min(), 'bottom': r.max(),
			                 'x200': lanes[k, 200], 'x120': lanes[k, 120], 'sigma': np.nanmean(sigma[k, r]) if np.isfinite(sigma[k, r]).any() else np.nan,
			                 'agreement': agreement(lanes[k], other), 'other_source_present': float(np.isfinite(other).sum() > 1),
			                 'other_side_present': float(np.isfinite(lanes[1 - k]).sum() > 1), 'width200': lanes[1, 200] - lanes[0, 200],
			                 'base_prob_mean': prob[k].mean(), 'base_prob_max': prob[k].max()})
			cands.append((k, s))
	lanes, conf, sigma = empty()
	if features:
		p = fusion['model'].predict_proba(pd.DataFrame(features)[fusion['columns']])[:, 1]
		for (k, s), pj in zip(cands, p):
			if pj > fusion['threshold'] and pj > conf[k]:
				lanes[k], conf[k], sigma[k] = sources[s][0][k], pj, sources[s][2][k]
	return lanes, conf, sigma