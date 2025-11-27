import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from structures import Fixation
import sqlite3

WORLD_TS_COL = "timestamp [ns]"          
FIX_START_COL = "start_timestamp [ns]"   # début fixation
FIX_END_COL = "end_timestamp [ns]"       # fin fixation
FIX_X_COL = "fixation x [px]"            # coordonnée x du regard 
FIX_Y_COL = "fixation y [px]"            # coordonnée y du regard 
FIX_X_IS_NORMALIZED = False              # True si x,y ∈ [0,1], False si déjà en pixels
DB_PATH = "database.sqlite"
WORLD_TS = "world_timestamps"


def load_from_db(db_path: str, cols: List[str], table: str) -> np.ndarray:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cols_escaped = ', '.join([f'"{col}"' for col in cols])
    
    cursor.execute(f"""
        SELECT
            {cols_escaped}
        FROM {table}""")
    
    rows = cursor.fetchall()
    arr = np.array(rows)
    conn.close()

    return arr


# Usage Example
# print("Chargement des fixations depuis la BDD...")
# fixation_data = load_from_db(DB_PATH, 
#                              [FIX_START_COL, FIX_END_COL, FIX_X_COL, FIX_Y_COL], 
#                              "fixations")