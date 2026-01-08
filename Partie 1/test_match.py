import match_images
import os 
import cv2
import numpy as np
from ptsInteretFixations import *
path_posters = "./data/Affiches/"
from create_projected_gaze import projected_gaze
from appelsDB import load_from_db
from heat_map import *
     
if __name__ == "__main__":
    images = []
    for filename in os.listdir(path_posters):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(path_posters, filename)
            img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
    n = 3000            
    res = ORB_on_fixations("./data/sujet2_f-835bf855", db_path="./data/database2.sqlite", video_filename="b7bd6c34_0.0-271.583.mp4",
                           nb_f=n,nb_min_kp=1500)
    ind_alea = np.random.randint(0,len(res))
    #ind_alea = 21
    print(ind_alea)
    image_test = res[ind_alea]["frame"]
    h, w = image_test.shape[:2]
    max_size = 800
    scale = max_size/max(h, w)
    img_display = cv2.resize(image_test, (int(w * scale), int(h * scale)))
    cv2.imshow("Image test", img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    kp_desc_all_posters = [match_images.apply_orb(img,n) for img in images]
    id_best_match, H = match_images.match_all(images,image_test,kp_desc_all_posters,20,n)
    print("Poster retrouvé:", os.listdir(path_posters)[id_best_match] if id_best_match!=-1 else "Aucun")

    # if id_best_match != -1:
    #     # load gazes from DB which match the fixation id in res[ind_alea]
    #     fixation_id = res[ind_alea]["fix_index"]
    #     arr = load_from_db(
    #         db_path="./data/database1.sqlite",
    #         cols=["gaze x [px]", "gaze y [px]"],
    #         table="gaze",
    #         where_clause='"fixation id" = ?',
    #         where_params=[fixation_id],
    #     )
    #     arr = projected_gaze(arr, H, clip=True, poster_size=(images[id_best_match].shape[1], images[id_best_match].shape[0]))
    #     h, w = images[id_best_match].shape[:2]
    #     x = arr[:,0]
    #     y = arr[:,1]
    #     y_plotly = h - y

    #     show_points_on_poster(f'./data/Affiches/{os.listdir(path_posters)[id_best_match]}', x, y_plotly)
    #     z = heat_map_density(x, y, w, h)
    #     z = z[::-1, :]
    #     show_heat_map_on_poster(f'./data/Affiches/{os.listdir(path_posters)[id_best_match]}', z)
