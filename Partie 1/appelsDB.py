import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Sequence, Tuple, Any
from structures import Fixation
import sqlite3

WORLD_TS_COL = "timestamp [ns]"          
FIX_START_COL = "start timestamp [ns]"   # début fixation
FIX_END_COL = "end timestamp [ns]"       # fin fixation
FIX_X_COL = "fixation x [px]"            # coordonnée x du regard 
FIX_Y_COL = "fixation y [px]"            # coordonnée y du regard 
FIX_X_IS_NORMALIZED = False              # True si x,y ∈ [0,1], False si déjà en pixels
DB_PATH = "database.sqlite"
WORLD_TS = "world_timestamps"

def load_from_db(
    db_path: str,
    cols: List[str],
    table: str,
    where_clause: Optional[str] = None,
    where_params: Optional[Sequence[Any]] = None,
) -> np.ndarray:
    """Load selected columns from table, optionally filtered by where_clause.

    - cols: list of column names (validated).
    - where_clause: SQL fragment to go after WHERE (use ? placeholders for params).
      Example: 'id = ? AND status = ?' or 'name LIKE ?'
    - where_params: sequence of parameter values matching the placeholders.
    Returns a numpy.ndarray with shape (n_rows, len(cols)).
    """
    # validate/escape identifiers (columns and table)
    cols_escaped = ', '.join([f'"{col}"' for col in cols])
    table_escaped = table

    sql = f"SELECT {cols_escaped} FROM {table_escaped}"
    params: Tuple[Any, ...] = ()
    if where_clause:
        sql += " WHERE " + where_clause
        params = tuple(where_params or ())

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
    finally:
        conn.close()

    # numpy array
    arr = np.array(rows, dtype=object)
    return arr


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


# Usage Example
# print("Chargement des fixations depuis la BDD...")
# fixation_data = load_from_db(DB_PATH, 
#                              [FIX_START_COL, FIX_END_COL, FIX_X_COL, FIX_Y_COL], 
#                              "fixations")
