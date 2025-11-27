#!/usr/bin/env python3
import sqlite3
import cv2
import numpy as np

def load_gaze_from_sqlite(db_path, table="gaze"):
    """Load gaze samples from SQLite DB."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Modify column names if your schema differs (Neon: “gaze x [px]”)
    cursor.execute(f"""
        SELECT
            "timestamp [ns]" AS ts,
            "gaze x [px]" AS x,
            "gaze y [px]" AS y
        FROM {table}
        ORDER BY ts ASC;
    """)

    rows = cursor.fetchall()
    conn.close()

    # Convert to arrays
    timestamps = np.array([r[0] for r in rows], dtype=np.int64)
    xs         = np.array([r[1] for r in rows], dtype=float)
    ys         = np.array([r[2] for r in rows], dtype=float)

    return timestamps, xs, ys


def find_gaze_for_frame(frame_time_ns, gaze_ts, xs, ys, tolerance_ns=10_000_000):
    """
    Find the nearest gaze sample to the frame timestamp.
    Tolerance default = 25ms.
    Returns (x, y) or None.
    """
    idx = np.searchsorted(gaze_ts, frame_time_ns)

    # Find nearest of idx or idx-1
    candidates = []
    if idx > 0:
        candidates.append(idx - 1)
    if idx < len(gaze_ts):
        candidates.append(idx)

    best = None
    best_dist = float("inf")

    for i in candidates:
        dist = abs(gaze_ts[i] - frame_time_ns)
        if dist < best_dist:
            best = i
            best_dist = dist

    if best_dist <= tolerance_ns:
        return float(xs[best]), float(ys[best])
    else:
        return None


def annotate_video(input_video, output_video, db_path):
    print(f"Loading gaze from: {db_path}")
    gaze_ts, xs, ys = load_gaze_from_sqlite(db_path)
    min_gaze_ts = np.min(gaze_ts)
    gaze_ts = gaze_ts - min_gaze_ts

    print(f"Gaze samples: {len(gaze_ts)}")

    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    dt_ns = 1e9 / fps  # nanoseconds per frame

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    frame_idx = 0
    print("Annotating video...")

    while frame_idx < 1000:
        ret, frame = cap.read()
        if not ret:
            break

        # Compute this frame's timestamp
        frame_time_ns = int(frame_idx * dt_ns)

        gaze = find_gaze_for_frame(frame_time_ns, gaze_ts, xs, ys)
        if gaze is not None:
            gx, gy = gaze
            gx = int(gx)
            gy = int(gy)

            # Draw gaze point (customize: size, color, thickness)
            cv2.circle(frame, (gx, gy), 12, (0, 0, 255), -1)
            cv2.circle(frame, (gx, gy), 24, (0, 0, 255), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Done! Annotated video saved as: {output_video}")


if __name__ == "__main__":
    #import argparse
    #parser = argparse.ArgumentParser(description="Overlay gaze points onto MP4 using SQLite DB.")
    #parser.add_argument("input_video", help="Path to input .mp4 scene video")
    #parser.add_argument("db_path", help="Path to SQLite database")
    #parser.add_argument("output_video", help="Path for output annotated video")
    #args = parser.parse_args()

    input_video = "./AcquisitionsEyeTracker/sujet1_f-42e0d11a/e0b2c246_0.0-138.011.mp4"
    output_video = "overlay.mp4"
    db_path = "database.sqlite"

    annotate_video(input_video, output_video, db_path)
