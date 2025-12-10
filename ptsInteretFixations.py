import cv2
import time
import numpy as np
from appelsDB import *
from undistort import undistort_frame, load_camera_calibration, undistort_points

def SIFT_on_fixations(
    data_folder: str,
    db_path: str,
    video_filename: str = "e0b2c246_0.0-138.011.mp4",
    table: str = "fixations",
    world_table: str = WORLD_TS,
    crop_size: int = 1500,
):
    """
    Pour chaque fixation dans la base de données, extraire un crop autour du point de fixation
    dans la vidéo undistordue, puis appliquer SIFT pour détecter des keypoints et des descripteurs.
    Retourne une liste de dictionnaires contenant les résultats pour chaque fixation :
    {
        "fix_index": index de la fixation,
        "frame_num": numéro de la frame dans la vidéo,
        "keypoints": liste des keypoints SIFT détectés,
        "descriptors": descripteurs SIFT associés aux keypoints,
    }
    """
    # Charger les fixations et timestamp de référence
    fixations = load_from_db(db_path, [FIX_START_COL, FIX_END_COL, FIX_X_COL, FIX_Y_COL], table)

    world_timestamps = load_from_db(db_path, [WORLD_TS_COL], world_table)
    reference_timestamp = float(world_timestamps[0][0])

    # Charger la calibration
    camera_file = f"{data_folder}/scene_camera.json"
    K, D = load_camera_calibration(camera_file)

    # Ouvrir la vidéo
    video_path = f"{data_folder}/{video_filename}"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo : {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)

    results = []
    for i, fixation in enumerate(fixations):
        #if i>=200 : break # To test
        start_ts = float(fixation[0]) - reference_timestamp
        end_ts = float(fixation[1]) - reference_timestamp
        fix_x = float(fixation[2])
        fix_y = float(fixation[3])

        mid_ts = (start_ts + end_ts) / 2.0
        mid_frame_num = int((mid_ts / 1e9) * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_num)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Undistort frame et point
        und_frame = undistort_frame(frame, K, D)
        und_pt = undistort_points([(fix_x, fix_y)], K, D)[0]

        # Extraire un crop autour du point (coordonnées sur l'image undistorted)
        cx, cy = int(und_pt[0]), int(und_pt[1])
        half = crop_size // 2
        h, w = und_frame.shape[:2]
        x0 = max(0, cx - half); x1 = min(w, cx + half)
        y0 = max(0, cy - half); y1 = min(h, cy + half)
        crop = und_frame[y0:y1, x0:x1].copy()

        # Appliquer SIFT sur crop avec OpenCV
        sift = cv2.ORB.create(nfeatures=2000)
        keypoints, descriptors = sift.detectAndCompute(crop, None)
        #print(f"Fixation {i}: {len(keypoints)} keypoints détectés.")

        if len(keypoints) <= 1500:
            # Ignore this fixation if not enough keypoints found
            # cv2.rectangle(und_frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
            # cv2.circle(und_frame, (cx, cy), 10, (0, 0, 255), 2)
            # cv2.imshow("Undistorted Frame", und_frame)
            # cv2.waitKey(0)
            pass
        else:
            entry = {
                "fix_index": i,
                "frame" : crop,
                "und_frame" : und_frame,
                "frame_num": mid_frame_num,
                "keypoints": keypoints,
                "descriptors": descriptors,
            }
        
            results.append(entry)

    cap.release()
    return results

if __name__ == "__main__":
    res = SIFT_on_fixations("./data/sujet2_f-835bf855", db_path="./data/sujet2_f-835bf855/database2.sqlite", video_filename="b7bd6c34_0.0-271.583.mp4")
    print(f"Traitement terminé : {len(res)} fixations traitées")