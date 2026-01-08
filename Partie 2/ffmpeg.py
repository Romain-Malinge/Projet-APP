import subprocess
from pathlib import Path

# ====== CONFIG ======
FFMPEG_PATH = r"C:/ffmpeg/bin/ffmpeg.exe"
VIDEO_PATH = r".\Partie 2\data\2025-11-20_15-30-11-a3a383b4\95cbe6dd_0.0-323.503.mp4"
OUTPUT_DIR = r".\Partie 2\videos\velo_1"

FPS = 30          # ex: 10 pour limiter à 10 fps, None pour tout
START_FRAME = 94   # frame de départ (0 = début)
END_FRAME = 500     # frame de fin (None = jusqu'à la fin)

START_NUMBER = 0
QUALITY = 2
# ====================

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Construction de la commande
cmd = [
    FFMPEG_PATH,
    "-i", VIDEO_PATH,
]

# Construction du filtre vidéo
filters = []

if FPS is not None:
    filters.append(f"fps={FPS}")

if END_FRAME is not None:
    filters.append(f"select='between(n,{START_FRAME},{END_FRAME})'")
else:
    filters.append(f"select='gte(n,{START_FRAME})'")

# Important : réindexer les timestamps
filters.append("setpts=N/FRAME_RATE/TB")

cmd += ["-vf", ",".join(filters)]

cmd += [
    "-q:v", str(QUALITY),
    "-start_number", str(START_NUMBER),
    str(Path(OUTPUT_DIR) / "%05d.jpg")
]

print("Commande FFmpeg :")
print(" ".join(cmd))

process = subprocess.run(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

if process.returncode != 0:
    print("❌ Erreur FFmpeg")
    print(process.stderr)
else:
    print("✅ Extraction terminée avec succès")
