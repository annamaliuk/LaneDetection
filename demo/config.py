import os
import numpy as np

DEMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models")
OUT_DIR = os.path.join(DEMO_DIR, "outputs")
EXAMPLE = os.path.join(DEMO_DIR, "example.mp4")
os.makedirs(OUT_DIR, exist_ok=True)

W, H = 640, 224
CROP = 0.4
ROWS = np.linspace(10, H - 1, 36).astype(int)