import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from ptsInterretPosterImages import load_posters

VIDEO_PATH = "data/AcquisitionsEyeTracker/sujet_f-42e0d11a/e0b2c246_0.0-138.011.mp4"
WORLD_TS_CSV = "data/AcquisitionsEyeTracker/sujet_f-42e0d11a/world_timestamps.csv"
FIXATIONS_CSV = "data/AcquisitionsEyeTracker/sujet_f-42e0d11a/fixations.csv"
POSTER_DIR = "data/Affiches"
OUTPUT_DETECTIONS_CSV = "output/poster_detections.csv"

# Taille de la ROI autour du regard (en pixels)
ROI_WIDTH = 400
ROI_HEIGHT = 400



def detect_posters_in_video():
    
    Path(OUTPUT_DETECTIONS_CSV).parent.mkdir(parents=True, exist_ok=True) #dossier de sortie si necessaire

    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # 1) Posters
    posters = load_posters(POSTER_DIR, sift)
    if not posters:
        print("[ERROR] Aucun poster chargé, vérifie POSTER_DIR")
        return

    # Afichage des postetrs charges avec par dessus les points d'interet detectes
    for poster in posters:
        img_path = Path(POSTER_DIR) / poster.name
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


if __name__ == "__main__":
    detect_posters_in_video()
