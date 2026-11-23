import subprocess, imageio_ffmpeg
from config import DEMO_DIR

subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-i', f"{DEMO_DIR}/05081544_0305.mp4", '-t', '30',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an', f"{DEMO_DIR}/example.mp4"], check=True)
print('saved', f"{DEMO_DIR}/example.mp4")