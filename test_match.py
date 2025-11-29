import match_images
import os 
import cv2
    
path_posters = "./data/Affiches/"
path_test = "./data/Affiches/Franklin.png"
     
if __name__ == "__main__":
    images = []
    for filename in os.listdir(path_posters):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(path_posters, filename)
            img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
             
    image_test = cv2.imread(path_test,cv2.IMREAD_GRAYSCALE)
    h, w = image_test.shape[:2]
    max_size = 800
    scale = max_size/max(h, w)
    img_display = cv2.resize(image_test, (int(w * scale), int(h * scale)))
    cv2.imshow("Image test", img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # for i in range(len(images)):
    #     match_images.match_and_display(images[i],image_test)
    sift_all_orig = [match_images.apply_sift(im) for im in images]
    sift_test = match_images.apply_sift(image_test)
    descriptors_all_orig = [desc for _, desc in sift_all_orig]
    descriptors_test = sift_test[1]
    id_best_match = match_images.match_all(descriptors_all_orig, descriptors_test, seuil=40)
    print("ID best match:", id_best_match)