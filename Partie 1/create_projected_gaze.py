import cv2
import numpy as np
import os
import match_images
# from create_projected_gaze import projected_gaze
from ptsInteretFixations import *
from appelsDB import load_from_db
from heat_map import *

def projected_gaze(gaze_of_frame, homography, clip = True, poster_size = None): 
    """
    Given a frame and a homography matrix, project the gaze points onto the poster plane, using cv2.perspectiveTransform.
    Input:  
        gaze_of_frame: list of (x,y) gaze points in the frame coordinate system
        homography: 3x3 homography matrix from frame to poster plane
        clip: whether to clip the projected points to the poster size
        poster_size: (width, height) of the poster, used for clipping if clip is True
    Output:
        projected_gaze_points: np array of (x,y) gaze points in the poster coordinate system
    """
    if homography is None or len(gaze_of_frame) == 0:
        return []

    gaze_points = np.array(gaze_of_frame, dtype=np.float32).reshape(-1, 1, 2)
    projected_points = cv2.perspectiveTransform(gaze_points, homography)
    projected_points = projected_points.reshape(-1, 2)
    projected_gaze_points = projected_points.astype(np.int16)

    if clip and poster_size is not None:
        w, h = poster_size
        projected_gaze_points[:, 0] = np.clip(projected_gaze_points[:, 0], 0, w - 1)
        projected_gaze_points[:, 1] = np.clip(projected_gaze_points[:, 1], 0, h - 1)

    return projected_gaze_points

if __name__ == "__main__":
    path_posters = "./data/Affiches/"
    images = []
    for filename in os.listdir(path_posters):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(path_posters, filename)
            img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                
    res = ORB_on_fixations("./data/sujet1_f-42e0d11a", db_path="./data/database1.sqlite", video_filename="e0b2c246_0.0-138.011.mp4")
    ind_alea = np.random.randint(0,len(res))
    # ind_alea = 263
    print(ind_alea)
    image_test = res[ind_alea]["frame"]
    h, w = image_test.shape[:2]
    max_size = 800
    scale = max_size/max(h, w)
    img_display = cv2.resize(image_test, (int(w * scale), int(h * scale)))
    cv2.imshow("Image test", img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    kp_desc_all_posters = [match_images.apply_orb(img) for img in images]
    id_best_match, H = match_images.match_all(images,image_test,kp_desc_all_posters)
    print("Poster retrouvé:", os.listdir(path_posters)[id_best_match] if id_best_match!=-1 else "Aucun")

    if id_best_match != -1:
        # load gazes from DB which match the fixation id in res[ind_alea]
        fixation_id = res[ind_alea]["fix_index"]
        arr = load_from_db(
            db_path="./data/database1.sqlite",
            cols=["gaze x [px]", "gaze y [px]"],
            table="gaze",
            where_clause='"fixation id" = ?',
            where_params=[fixation_id],
        )

        arr = projected_gaze(arr, H, clip=True, poster_size=(images[id_best_match].shape[1], images[id_best_match].shape[0]))
        h, w = images[id_best_match].shape[:2]
        x = arr[:,0]
        y = arr[:,1]
        y_plotly = h - y

        show_points_on_poster(f'./data/Affiches/{os.listdir(path_posters)[id_best_match]}', x, y_plotly)
