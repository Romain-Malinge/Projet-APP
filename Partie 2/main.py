import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from convert_to_sql import csv_to_sqlite
from appelsDB import *
import time
from plotting import plot_segmentation_non_blocking, show_segmentation_opencv

WORKING_DIR = "data"
SUJET_NAMES = ["2025-11-20_15-30-11-a3a383b4", "2025-11-20_15-40-17-10b70589"]
VIDEO_FILENAMES = ["95cbe6dd_0.0-323.503.mp4", "3238656b_0.0-265.674.mp4"]

sujet_id = 0
start_frame = 65
pas_frame = 10

DB_PATH = f"{WORKING_DIR}/database{sujet_id+1}.sqlite"
positions_all_posters = dict() # le resultat final

# Création de la DB SQLite si elle n'existe pas déjà
if not os.path.exists(DB_PATH):
    print(f"[INFO] Création de la DB SQLite pour le sujet {sujet_id+1}...")
    print(f"{WORKING_DIR}/{SUJET_NAMES[sujet_id]}",DB_PATH)
    csv_to_sqlite(f"{WORKING_DIR}/{SUJET_NAMES[sujet_id]}", DB_PATH, False)

# Load model
processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic", use_fast=True)
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")

def segmentation_from_frame(pil_image, verbose=False):
    start_time = time.time()
    inputs = processor(images=pil_image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    segmentation = processor.post_process_semantic_segmentation(outputs, target_sizes=[(height, width)])[0]
    seg_np = segmentation.cpu().numpy().astype(np.uint8)
    if verbose:
        end_time = time.time()
        print(f"Segmentation effectuée en {end_time - start_time:.2f} secondes")
    return seg_np

# Données gaze etc
world_timestamps = load_from_db(DB_PATH, [WORLD_TS_COL], WORLD_TS)
reference_timestamp = float(world_timestamps[0][0])
# Charger les fixations et timestamp de référence
fixations = load_from_db(DB_PATH, [FIX_START_COL, FIX_END_COL, FIX_X_COL, FIX_Y_COL, "fixation id"], "fixations")

# plt.ion()  # Mode interactif ON
# fig, ax = plt.subplots(figsize=(15, 10))

# Ouvrir vidéo
video_path = f"{WORKING_DIR}/{SUJET_NAMES[sujet_id]}/{VIDEO_FILENAMES[sujet_id]}"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Impossible d'ouvrir la vidéo : {video_path}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Démarrage frame {start_frame} sur {total_frames} frames totales")

# Positionner au start_frame
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

frame_count = start_frame
processed_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fin de vidéo")
            break
        
        # Traitement frame
        if frame_count % pas_frame == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            seg_np = segmentation_from_frame(pil_image, True)
            
            # plot_segmentation_non_blocking(ax, rgb_frame, seg_np, frame_count)
            show_segmentation_opencv(rgb_frame, seg_np)
            
            print(f"Frame {frame_count} traitée")
        
        frame_count += 1
        
        # Délai optionnel pour contrôler la vitesse
        # plt.pause(0.1)  # 100ms entre frames (décommente si trop rapide)

except KeyboardInterrupt:
    print("Arrêt par utilisateur (Ctrl+C)")

cap.release()
# plt.ioff()
# plt.show(block=True)  # Garde la dernière image
print(f"Traitement terminé : frames {start_frame} à {frame_count-1}")