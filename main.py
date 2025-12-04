import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from ptsInteretPosterImages import load_posters

WORKING_DIR = "data"
SUJET_NAMES = ["sujet1_f-42e0d11a", "sujet2_f-835bf855", "sujet3_m-84ce1158", "sujet4_m-fee537df", "sujet5_m-671cf44e", "sujet6_m-0b355b51"]
POSTERS_DIR = f"{WORKING_DIR}/Affiches"
OUTPUT_DETECTIONS_CSV = "output/poster_detections.csv"

# Taille de la ROI autour du regard (en pixels)
ROI_WIDTH = 400
ROI_HEIGHT = 400



def detect_posters_in_video(display = False, sujet_index: int = 0):
    
    Path(OUTPUT_DETECTIONS_CSV).parent.mkdir(parents=True, exist_ok=True) #dossier de sortie si necessaire

    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

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

    # 3) Heatmap


if __name__ == "__main__":
    detect_posters_in_video()
