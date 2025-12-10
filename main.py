import cv2
import os
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from ptsInteretPosterImages import load_posters
from convert_to_sql import csv_to_sqlite
from ptsInteretFixations import SIFT_on_fixations

WORKING_DIR = "data"
SUJET_NAMES = ["sujet1_f-42e0d11a", "sujet2_f-835bf855", "sujet3_m-84ce1158", "sujet4_m-fee537df", "sujet5_m-671cf44e", "sujet6_m-0b355b51"]
VIDEO_FILENAMES = ["e0b2c246_0.0-138.011.mp4", "b7bd6c34_0.0-271.583.mp4", "422f10f2_0.0-247.734.mp4", "2fb8301a_0.0-71.632.mp4", "585d8df7_0.0-229.268.mp4", "429d311a_0.0-267.743.mp4"]
POSTERS_DIR = f"{WORKING_DIR}/Affiches"
OUTPUT_DETECTIONS_CSV = "output/poster_detections.csv"

# Taille de la ROI autour du regard (en pixels)
ROI_WIDTH = 400
ROI_HEIGHT = 400



def detect_posters_in_video(display = False, sujet_index: int = 0):
    
    Path(OUTPUT_DETECTIONS_CSV).parent.mkdir(parents=True, exist_ok=True) #dossier de sortie si necessaire

    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    db_path = f"{WORKING_DIR}/database{sujet_index+1}.sqlite"

    # Création de la DB SQLite si elle n'existe pas déjà
    if not os.path.exists(db_path):
        print(f"[INFO] Création de la DB SQLite pour le sujet {sujet_index+1}...")
        csv_to_sqlite(f"{WORKING_DIR}/{SUJET_NAMES[sujet_index]}", db_path, display)

    # 1) Posters
    posters = load_posters(POSTERS_DIR, sift)
    if not posters:
        print(f"[ERROR] Aucun poster chargé, vérifie {POSTERS_DIR}")
        return
    
    if display:
        # Affichage des posters chargés avec par dessus les points d'intérêt détectés
        for poster in posters:
            img_path = Path(POSTERS_DIR) / poster.name
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            img_kp = cv2.drawKeypoints(img, poster.kp, None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            #img_kp = cv2.drawKeypoints(img, poster.kp, None)  # sans flags

            h, w = img_kp.shape[:2]
            max_size = 800  # taille max pour l’affichage
            scale = max_size / max(h, w)
            if scale < 1.0:
                img_kp_display = cv2.resize(img_kp, (int(w * scale), int(h * scale)))
            else:
                img_kp_display = img_kp

            cv2.imshow(f"Poster: {poster.name}", img_kp_display)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 2) Vidéo et fixations
    res = SIFT_on_fixations(
        data_folder=f"{WORKING_DIR}/{SUJET_NAMES[sujet_index]}",
        db_path=db_path,
        video_filename=f"{VIDEO_FILENAMES[sujet_index]}")
    
    # 
    # 3) Heatmap


if __name__ == "__main__":
    detect_posters_in_video()
