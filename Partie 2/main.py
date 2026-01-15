import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from convert_to_sql import csv_to_sqlite
from appelsDB import *
import time

WORKING_DIR = "data"
SUJET_NAMES = ["2025-11-20_15-30-11-a3a383b4", "2025-11-20_15-40-17-10b70589"]
VIDEO_FILENAMES = ["95cbe6dd_0.0-323.503.mp4", "3238656b_0.0-265.674.mp4"]

sujet_id = 0
start_frame = 65

sujet_id = 0
start_frame = 1000 

DB_PATH = f"{WORKING_DIR}/database{sujet_id+1}.sqlite"
positions_all_posters = dict() # le resultat final

# Création de la DB SQLite si elle n'existe pas déjà
if not os.path.exists(DB_PATH):
    print(f"[INFO] Création de la DB SQLite pour le sujet {sujet_id+1}...")
    print(f"{WORKING_DIR}/{SUJET_NAMES[sujet_id]}",DB_PATH)
    csv_to_sqlite(f"{WORKING_DIR}/{SUJET_NAMES[sujet_id]}", DB_PATH, False)

def plot_segmentation_non_blocking(ax, original, segmentation, frame_num):
    """Version non-blocante de plot_segmentation"""
    cityscapes_colors = {
        0: (128, 64, 128), 1: (244, 35, 232), 2: (70, 70, 70),
        3: (102, 102, 156), 4: (190, 153, 153), 5: (153, 153, 153),
        6: (250, 170, 30), 7: (220, 220, 0), 8: (107, 142, 35),
        9: (152, 251, 152), 10: (70, 130, 180), 11: (220, 20, 60),
        12: (255, 0, 0), 13: (0, 0, 142), 14: (0, 0, 70),
        15: (0, 60, 100), 16: (0, 80, 100), 17: (0, 0, 230), 18: (119, 11, 32)
    }
    
    cityscapes_labels = {
        0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence', 
        5: 'pole', 6: 'traffic light', 7: 'traffic sign', 8: 'vegetation', 
        9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider', 13: 'car', 
        14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle'
    }
    
    # Clear previous plot
    ax.clear()
    
    # Segmentation RGB overlay
    segmentation_rgb = np.zeros((*segmentation.shape, 3), dtype=np.uint8)
    for class_id, color in cityscapes_colors.items():
        segmentation_rgb[segmentation == class_id] = color
    
    # Affichage avec overlay
    ax.imshow(original)
    ax.imshow(segmentation_rgb, alpha=0.5)
    
    ax.set_title(f'Semantic Segmentation - Frame {frame_num}')
    ax.axis('off')
    
    # Legend dynamique (seulement classes présentes)
    patches = []
    unique_classes = np.unique(segmentation)
    for class_id in unique_classes:
        if class_id in cityscapes_labels:
            color_tuple = tuple(c / 255.0 for c in cityscapes_colors[class_id])
            patches.append(mpatches.Patch(color=color_tuple, label=f'{cityscapes_labels[class_id]}'))
    
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.draw()
    plt.pause(0.01)  # Pause très courte pour rafraîchir

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

# Load model
processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")

# Données gaze etc
world_timestamps = load_from_db(DB_PATH, [WORLD_TS_COL], WORLD_TS)
reference_timestamp = float(world_timestamps[0][0])
# Charger les fixations et timestamp de référence
fixations = load_from_db(DB_PATH, [FIX_START_COL, FIX_END_COL, FIX_X_COL, FIX_Y_COL, "fixation id"], "fixations")

plt.ion()  # Mode interactif ON
fig, ax = plt.subplots(figsize=(15, 10))

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
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        seg_np = segmentation_from_frame(pil_image, True)
        
        plot_segmentation_non_blocking(ax, rgb_frame, seg_np, frame_count)
        
        print(f"Frame {frame_count} traitée")
        
        frame_count += 1
        
        # Délai optionnel pour contrôler la vitesse
        # plt.pause(0.1)  # 100ms entre frames (décommente si trop rapide)

except KeyboardInterrupt:
    print("Arrêt par utilisateur (Ctrl+C)")

cap.release()
plt.ioff()
plt.show(block=True)  # Garde la dernière image
print(f"Traitement terminé : frames {start_frame} à {frame_count-1}")
