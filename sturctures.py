import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

@dataclass
class PosterRef:
    name: str
    kp: Any
    des: np.ndarray
    size: tuple  # (w, h)


@dataclass
class Fixation:
    fixation_id: int
    start: float
    end: float
    x: float
    y: float


@dataclass
class FixationFrameCandidate:
    fixation_id: int
    frame_idx: int
    x: float
    y: float


@dataclass
class PosterDetection:
    fixation_id: int
    frame_idx: int
    timestamp: float
    poster_name: str
    inliers: int
    total_matches: int
    inlier_ratio: float
    x: float
    y: float
  