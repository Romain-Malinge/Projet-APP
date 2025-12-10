import match_images
import os 
import cv2
import numpy as np
from ptsInteretFixations import SIFT_on_fixations
path_posters = "./data/Affiches/"
     
if __name__ == "__main__":
    images = []
    for filename in os.listdir(path_posters):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(path_posters, filename)
            img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                
    res = SIFT_on_fixations("./data/sujet1_f-42e0d11a", video_filename="e0b2c246_0.0-138.011.mp4")
    ind_alea = np.random.randint(0,len(res))
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
    id_best_match = match_images.match_all(images,image_test,kp_desc_all_posters)
    print("Poster retrouvé:", os.listdir(path_posters)[id_best_match] if id_best_match!=-1 else "Aucun")