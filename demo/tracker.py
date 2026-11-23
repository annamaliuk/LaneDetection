import numpy as np
from config import W, H

Q_POS, Q_VEL, R0, GATE, MAX_COAST = 0.5, 0.1, 9.0, 11.34, 30
ROWS3 = np.array([215, 150, 85])
F = np.eye(6)
F[:3, 3:] = np.eye(3)


def measure(lane, conf, sigma):
	rows = np.where(np.isfinite(lane))[0]
	if len(rows) < 8 or rows.max() < 150:
		return None
	z = np.polyval(np.polyfit(rows, lane[rows], 2), ROWS3)
	if np.isfinite(sigma[rows]).any():
		R = np.diag(np.interp(ROWS3, rows, np.nan_to_num(sigma[rows], nan=5.0)) ** 2 + 1)
	else:
		R = np.eye(3) * R0 / max(conf, 0.05)
	return {'z': z, 'R': R, 'top': rows.min()}


class Track:
	def __init__(self, m, confirmed):
		self.x = np.r_[m['z'], 0, 0, 0].astype(float)
		self.P = np.diag([25.0, 25, 25, 4, 4, 4])
		self.top, self.confirmed = m['top'], confirmed
		self.hits, self.misses, self.rejects, self.q = 1, 0, 0, 1.0

	def predict(self):
		self.x = F @ self.x
		self.P = F @ self.P @ F.T + np.diag([Q_POS] * 3 + [Q_VEL] * 3) * self.q

	def distance(self, m):
		y = m['z'] - self.x[:3]
		return y @ np.linalg.solve(self.P[:3, :3] + m['R'], y)

	def update(self, m):
		K = self.P[:, :3] @ np.linalg.inv(self.P[:3, :3] + m['R'])
		self.x = self.x + K @ (m['z'] - self.x[:3])
		self.P = self.P - K @ self.P[:3]
		self.top = m['top']

	def curve(self):
		out = np.full(H, np.nan, np.float32)
		out[self.top:] = np.polyval(np.polyfit(ROWS3, self.x[:3], 2), np.arange(self.top, H))
		return out


class Kalman:
	def __init__(self):
		self.tracks = [None, None]
		self.width = None

	def step(self, lanes, conf, sigma):
		for t in self.tracks:
			if t:
				t.predict()
		m = [measure(lanes[k], conf[k], sigma[k]) for k in range(2)]
		left, right = self.tracks
		if left and right and m[0] and m[1] and left.distance(m[1]) + right.distance(m[0]) < left.distance(m[0]) + right.distance(m[1]):
			m = m[::-1]
		real = [m[0] is not None, m[1] is not None]
		for k, sign in [(0, -1), (1, 1)]:
			if self.width is not None and m[k] is None and m[1 - k] and self.tracks[k]:
				m[k] = {'z': m[1 - k]['z'] + sign * self.width, 'R': m[1 - k]['R'] * 4, 'top': m[1 - k]['top']}
		for k in range(2):
			t = self.tracks[k]
			if m[k] is None:
				if t:
					t.misses += 1
					if t.misses > MAX_COAST or not t.confirmed:
						self.tracks[k] = None
			elif t is None:
				if real[k]:
					self.tracks[k] = Track(m[k], False)
			elif t.confirmed and t.distance(m[k]) > GATE:
				t.rejects, t.misses, t.q = t.rejects + 1, t.misses + 1, 4.0
				if t.rejects >= 3 and real[k]:
					self.tracks[k] = Track(m[k], True)
				elif t.misses > MAX_COAST:
					self.tracks[k] = None
			else:
				d2 = t.distance(m[k])
				t.update(m[k])
				t.rejects, t.misses, t.hits = 0, 0, t.hits + 1
				t.confirmed = t.confirmed or t.hits >= 3
				t.q = min(t.q * 2, 8.0) if d2 > GATE / 2 else max(t.q / 2, 1.0)
		left, right = self.tracks
		if left and right and all(real):
			w = right.x[:3] - left.x[:3]
			self.width = w if self.width is None else 0.9 * self.width + 0.1 * w
		if left and left.x[0] > W / 2:
			self.tracks = [None, left]
		elif right and right.x[0] < W / 2:
			self.tracks = [right, None]
		out = np.full((2, H), np.nan, np.float32)
		coasting = [False, False]
		for k, t in enumerate(self.tracks):
			if t and t.confirmed:
				out[k], coasting[k] = t.curve(), t.misses > 0
		return out, coasting