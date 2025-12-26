#!/usr/bin/env python3
import sqlite3
import cv2
import numpy as np

# Adjustable parameters -------------------------

FADE_FACTOR = 0.98
"""
How fast the heatmap fades each frame.
0.90 = fast fade
0.98 = slow fade
"""

HEAT_INTENSITY = 10.0
"""
How much "heat" each gaze point deposits.
"""

KERNEL_SIZE = 150
"""
Radius of the Gaussian blob added around gaze points.
"""

ALPHA = 0.7
"""
How strong the heatmap overlay appears on the video.
1.0 = strong red
0.3 = subtle overlay
"""

# ------------------------------------------------


def load_gaze_from_sqlite(db_path, table="gaze"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

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

    if not rows:
        raise RuntimeError("No gaze samples found in DB.")

    timestamps = np.array([r[0] for r in rows], dtype=np.int64)
    xs = np.array([r[1] for r in rows], dtype=float)
    ys = np.array([r[2] for r in rows], dtype=float)

    return timestamps, xs, ys


def make_gaussian_kernel(size, intensity=1.0):
    """Create a 2D Gaussian kernel."""
    ax = np.linspace(-(size / 2), size / 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * (size / 4)**2))
    kernel *= intensity
    return kernel


def annotate_video(input_video, output_video, db_path):
    print(f"Loading gaze data from {db_path} ...")
    gaze_ts, xs, ys = load_gaze_from_sqlite(db_path)
    min_gaze_ts = np.min(gaze_ts)
    gaze_ts = gaze_ts - min_gaze_ts

    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    dt_ns = 1e9 / fps

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    heatmap = np.zeros((height, width), dtype=np.float32)
    kernel = make_gaussian_kernel(KERNEL_SIZE, HEAT_INTENSITY)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    print("Generating progressive heatmap video...")

    frame_idx = 0
    gaze_index = 0
    total_gaze = len(xs)

    while frame_idx < 1000:
        ret, frame = cap.read()
        if not ret:
            break

        frame_ts = int(frame_idx * dt_ns)

        # DECAY existing heatmap
        heatmap *= FADE_FACTOR

        # ADD gaze points for this frame
        while gaze_index < total_gaze and gaze_ts[gaze_index] <= frame_ts:
            gx = int(xs[gaze_index])
            gy = int(ys[gaze_index])
            gaze_index += 1

            if gx <= 0 or gx >= width or gy <= 0 or gy >= height:
                continue

            # Add Gaussian kernel centered at gaze point
            k = kernel
            ks = KERNEL_SIZE
            r = ks // 2

            x1 = max(gx - r, 0)
            x2 = min(gx + r, width)
            y1 = max(gy - r, 0)
            y2 = min(gy + r, height)

            kx1 = r - (gx - x1)
            ky1 = r - (gy - y1)
            kx2 = kx1 + (x2 - x1)
            ky2 = ky1 + (y2 - y1)

            heatmap[y1:y2, x1:x2] += k[ky1:ky2, kx1:kx2]

        # Normalize heatmap for visualization
        hm_norm = heatmap.copy()
        if hm_norm.max() > 0:
            hm_norm = hm_norm / hm_norm.max()

        # Convert heatmap to red overlay
        heat_color = np.zeros((height, width, 3), dtype=np.uint8)
        heat_color[:, :, 2] = (hm_norm * 255).astype(np.uint8)  # Red channel

        overlay = cv2.addWeighted(frame, 1.0, heat_color, ALPHA, 0)

        out.write(overlay)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Done! Heatmap video saved to: {output_video}")


if __name__ == "__main__":
    #import argparse
    #parser = argparse.ArgumentParser(description="Overlay gaze points onto MP4 using SQLite DB.")
    #parser.add_argument("input_video", help="Path to input .mp4 scene video")
    #parser.add_argument("db_path", help="Path to SQLite database")
    #parser.add_argument("output_video", help="Path for output annotated video")
    #args = parser.parse_args()

    input_video = "../AcquisitionsEyeTracker/sujet1_f-42e0d11a/e0b2c246_0.0-138.011.mp4"
    output_video = "heatmap_overlay.mp4"
    db_path = "database.sqlite"

    annotate_video(input_video, output_video, db_path)
