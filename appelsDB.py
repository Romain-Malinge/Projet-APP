import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from sturctures import Fixation
import sqlite3

WORLD_TS_COL = "timestamp [ns]"          
FIX_START_COL = "start_timestamp [ns]"   # début fixation
FIX_END_COL = "end_timestamp [ns]"       # fin fixation
FIX_X_COL = "fixation x [px]"            # coordonnée x du regard 
FIX_Y_COL = "fixation y [px]"            # coordonnée y du regard 
FIX_X_IS_NORMALIZED = False              # True si x,y ∈ [0,1], False si déjà en pixels
DB_PATH = "database.sqlite"
WORLD_TS = "world_timestamps"


def load_world_timestamps_db(db_path: str) -> np.ndarray:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"""
        SELECT
            {WORLD_TS_COL}
        FROM {WORLD_TS}
        ORDER BY {WORLD_TS_COL} ASC;
                   """)
    
    rows = cursor.fetchall()
    conn.close()

    return rows

def load_fixations_db(db_path: str) -> List[Fixation]:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"""
        SELECT
            {FIX_START_COL},
            {FIX_END_COL},
            {FIX_X_COL},
            {FIX_Y_COL}
        FROM fixations
        ORDER BY {FIX_START_COL} ASC;
                   """)

    rows = cursor.fetchall()
    conn.close()

    fixations: List[Fixation] = []
    for i, row in enumerate(rows):
        fix = Fixation(
            fixation_id=i,
            start=float(row[0]),
            end=float(row[1]),
            x=float(row[2]),
            y=float(row[3]),
        )
        fixations.append(fix)

    return fixations
