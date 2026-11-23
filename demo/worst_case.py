import cv2, numpy as np, pandas as pd
from config import EXAMPLE, OUT_DIR
from video import process_video

cap = cv2.VideoCapture(EXAMPLE)
frames = [cap.read()[1] for _ in range(30)]
cases = {
	'night + noise': [np.clip(f * 0.25 + np.random.normal(0, 8, f.shape), 0, 255).astype(np.uint8) for f in frames],
	'black': [np.zeros((360, 640, 3), np.uint8)] * 30,
	'noise': [np.random.randint(0, 256, (360, 640, 3), np.uint8) for _ in range(30)],
	'single frame': frames[:1],
	'odd size 333x177': [cv2.resize(f, (333, 177)) for f in frames],
	'tiny 96x54': [cv2.resize(f, (96, 54)) for f in frames],
	'4k': [cv2.resize(f, (3840, 2160)) for f in frames[:5]],
}
results = []
for name, fs in cases.items():
	path = f"{OUT_DIR}/test_{name}.avi"
	writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), 25, fs[0].shape[1::-1])
	for f in fs:
		writer.write(f)
	writer.release()
	results.append({'case': name} | process_video(path)[1].iloc[0].to_dict())
for name in ['broken.mp4', 'missing.mp4']:
	open(f"{OUT_DIR}/broken.mp4", 'w').write('not a video')
	try:
		process_video(f"{OUT_DIR}/{name}")
	except ValueError as e:
		results.append({'case': name, 'frames': 0, 'error': str(e)})
table = pd.DataFrame(results)
table.to_csv(f"{OUT_DIR}/worst_case_test.csv", index=False)
print(table.to_string())