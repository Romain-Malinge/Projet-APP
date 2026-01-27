import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from convert_to_sql import csv_to_sqlite
from appelsDB import WORLD_TS_COL, WORLD_TS, load_from_db
import time
from plotting import plot_segmentation_non_blocking, show_segmentation_opencv
from chronique_temporelle import ChroniqueTemporelle
from track_gaze import find_gaze_for_frame
import math

WORKING_DIR = "data"
SUJET_NAMES = ["2025-11-20_15-30-11-a3a383b4", "2025-11-20_15-40-17-10b70589"]
VIDEO_FILENAMES = ["95cbe6dd_0.0-323.503.mp4", "3238656b_0.0-265.674.mp4"]

sujet_id = 0
# Variables de temps en s
start_temps = 25
pas_temps = 0.25

DB_PATH = f"{WORKING_DIR}/database{sujet_id + 1}.sqlite"
positions_all_posters = dict()  # le resultat final

# Création de la DB SQLite si elle n'existe pas déjà
if not os.path.exists(DB_PATH):
    print(f"[INFO] Création de la DB SQLite pour le sujet {sujet_id + 1}...")
    print(f"{WORKING_DIR}/{SUJET_NAMES[sujet_id]}", DB_PATH)
    csv_to_sqlite(f"{WORKING_DIR}/{SUJET_NAMES[sujet_id]}", DB_PATH, False)


def segmentation_from_frame(pil_image, verbose=False):
    start_time = time.time()
    inputs = processor(images=pil_image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    segmentation = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[(height, width)]
    )[0]
    seg_np = segmentation.cpu().numpy().astype(np.uint8)
    if verbose:
        end_time = time.time()
        print(f"Segmentation effectuée en {end_time - start_time:.2f} secondes")
    return seg_np


# Données gaze etc
world_timestamps = load_from_db(DB_PATH, [WORLD_TS_COL], WORLD_TS)
reference_timestamp = int(world_timestamps[0][0])
# Charger les fixations et timestamp de référence
gaze = load_from_db(DB_PATH, ["timestamp [ns]", "gaze x [px]", "gaze y [px]"], "gaze")
gaze_ts = np.array(gaze[:, 0], dtype=np.int64)
gaze_ts = gaze_ts - min(gaze_ts)
gaze_xs = np.array(gaze[:, 1], dtype=float)
gaze_ys = np.array(gaze[:, 2], dtype=float)

# Load model
processor = AutoImageProcessor.from_pretrained(
    "facebook/mask2former-swin-large-cityscapes-semantic", use_fast=True
)
model = Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-large-cityscapes-semantic"
)

# plt.ion()  # Mode interactif ON
# fig, ax = plt.subplots(figsize=(15, 10))

# Ouvrir vidéo
video_path = f"{WORKING_DIR}/{SUJET_NAMES[sujet_id]}/{VIDEO_FILENAMES[sujet_id]}"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Impossible d'ouvrir la vidéo : {video_path}")
fps = cap.get(cv2.CAP_PROP_FPS)
# video_length = math.floor(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
# print(video_length)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Init chronique temporelle
cityscapes_labels = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
}

regroupements = {
    0: 0,
    1: 0,
    2: 2,
    3: 2,
    4: 2,
    5: 7,
    6: 7,
    8: 8,
    9: 8,
    10: 10,
    11: 11,
    12: 18,
    13: 13,
    14: 13,
    15: 13,
    16: None,
    17: 18,
    18: 18,
}
new_labels = {
    0: "route",
    2: "batiment",
    7: "panneau",
    8: "vegetation",
    10: "ciel",
    11: "piéton",
    13: "4 roues",
    18: "2 roues",
}
chronique_temporelle = ChroniqueTemporelle(new_labels.values())

print(f"Démarrage frame {int(start_temps * fps)} sur {total_frames} frames totales")

# Positionner au start_frame
curr_time = start_temps
frame_count = 0

try:
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(curr_time * fps))
        ret, frame = cap.read()
        if not ret:
            print("Fin de vidéo")
            break

        # Traitement frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        seg_np = segmentation_from_frame(pil_image, True)

        gaze_x, gaze_y = find_gaze_for_frame(curr_time * 1e9, gaze_ts, gaze_xs, gaze_ys)

        # plot_segmentation_non_blocking(ax, rgb_frame, seg_np, frame_count)
        show_segmentation_opencv(rgb_frame, seg_np, gaze_x, gaze_y)

        label_id = int(seg_np[int(gaze_y), int(gaze_x)])

        if label_id in cityscapes_labels.keys():
            new_label = regroupements[label_id]
            if new_label is not None:
                chronique_temporelle.ajouter_frame(
                    new_labels[new_label], int(curr_time * fps)
                )

        print(f"Frame n°{int(curr_time * fps)} traitée : {cityscapes_labels[label_id]}")

        frame_count += 1
        curr_time = curr_time + pas_temps

        # Délai optionnel pour contrôler la vitesse
        # plt.pause(0.1)  # 100ms entre frames (décommente si trop rapide)

except KeyboardInterrupt:
    print("Arrêt par utilisateur (Ctrl+C)")

cap.release()
# plt.ioff()
# plt.show(block=True)  # Garde la dernière image
print("Traitement terminé")
chronique_temporelle.afficher()
