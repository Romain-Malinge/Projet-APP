import json
import numpy as np
import cv2

def load_camera_calibration(json_path):
    """Load camera matrix and distortion coeffs from scene_camera.json."""
    with open(json_path, "r") as f:
        data = json.load(f)

    camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
    dist_coeffs   = np.array(data["distortion_coefficients"], dtype=np.float64).reshape(-1)

    return camera_matrix, dist_coeffs


def undistort_fisheye_frame(frame, K, D):
    """
    Undistort a single frame using OpenCV fisheye model.
    Supports 8-coefficient distortion (as in Pupil Labs calibration).
    """
    h, w = frame.shape[:2]
    dim = (w, h)

    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, dim, 0.0, (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)

    # Apply remapping
    undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
    
    return undistorted

def undistort_video(camera_file, input_video, output_video, db_path):
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    dt_ns = 1e9 / fps

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    print("Generating undistorted video...")
    
    K, D = load_camera_calibration(camera_file)
    # Precompute mapping ONCE (much faster)
    ret, test_frame = cap.read()
    h, w = test_frame.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0.0, (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), cv2.CV_16SC2)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # restart video

    frame_idx = 0
    while frame_idx < 1000:
        frame_idx += 1
        ret, frame = cap.read()
        if not ret:
            break

        undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        out.write(undistorted)

    cap.release()
    out.release()
    print("Saved undistorted video ✔")


if __name__ == "__main__":
    camera_file = "../AcquisitionsEyeTracker/sujet1_f-42e0d11a/scene_camera.json"
    input_video = "../AcquisitionsEyeTracker/sujet1_f-42e0d11a/e0b2c246_0.0-138.011.mp4"
    output_video = "undistorted.mp4"
    db_path = "database.sqlite"

    undistort_video(camera_file, input_video, output_video, db_path)