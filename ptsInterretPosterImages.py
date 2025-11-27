
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from sturctures import PosterRef


def load_posters(poster_dir: str, sift: cv2.SIFT) -> List[PosterRef]:
    poster_dir = Path(poster_dir)
    posters: List[PosterRef] = []

    for img_path in poster_dir.glob("*"):
        if not img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Impossible de lire {img_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        if des is None or len(kp) == 0:
            print(f"Pas de features SIFT pour {img_path.name}")
            continue

        h, w = gray.shape
        posters.append(PosterRef(
            name=img_path.name,
            kp=kp,
            des=des,
            size=(w, h),
        ))
        print(f"Poster chargé : {img_path.name} ({len(kp)} keypoints)")

    return posters