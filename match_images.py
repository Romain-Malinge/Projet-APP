import cv2
import numpy as np


def apply_sift(image):
    # Image en niveaux de gris
    sift = cv2.SIFT.create(nfeatures=2000, contrastThreshold=0.03)
    keypoints,descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def match_keypoints(descriptors_orig, descriptors_video):
    """
    Calcule les correspondances entre un poster et une image de la vidéo
    descriptors_orig : descripteur de l'affiche originale
    descriptors_video : descripteur obtenue dans la video
    """
    # match selon la distance minimale entre descripteurs
    bf = cv2.BFMatcher(cv2.NORM_L2,crossCheck=False) # True -> contrainte de symétrie
    matches = bf.knnMatch(descriptors_orig,descriptors_video,k=2)
    good_matches = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance: # Ratio test 
            good_matches.append(m)
    return good_matches


def match_and_display(img1, img2, max_matches=50):
    """
    Détecte et affiche les correspondances SIFT entre deux images.
    img_path1 : chemin de la première image.
    img_path2 : chemin de la deuxième image.
    max_matches : nombre maximum de correspondances à afficher.
    """
    # Charger les images en niveaux de gris
    kp1,d1 = apply_sift(img1)
    kp2,d2 = apply_sift(img2)
    matches = match_keypoints(d1,d2)
    h, w = img1.shape[:2]
    max_size = 800  # taille max pour l’affichage
    scale = max_size/max(h, w)
    # Dessiner les correspondances
    matched_img = cv2.drawMatches(
        img1, kp1, img2, kp2, matches[:max_matches], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    img_display = cv2.resize(matched_img, (int(w * scale), int(h * scale)))
    cv2.imshow("SIFT Matches", img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
# Adapter seuil
def match_all(descriptors_all_orig,descriptors_video,seuil=80):
    """
    Fait les matches de l'image de la vidéo avec tous les posters
    et renvoie le poster avec le plus de matches (-1 si aucun ne convient)
    descriptors_all_orig : liste des descripteurs de toutes les affiches
    descriptos_video : descripteur de la vidéo
    seuil : seuil minimal pour identifier un poster
    """
    id_best_match = -1
    pourcentage_match_max = 0
    for i in range(len(descriptors_all_orig)):
        matches = match_keypoints(descriptors_all_orig[i],descriptors_video)
        pourcentage_match = 100*(len(matches)/max(len(descriptors_all_orig[i]),len(descriptors_video)))
        print(pourcentage_match)
        if pourcentage_match > seuil:
            # faire homographie + match à nouveau pour vérif
            if pourcentage_match > pourcentage_match_max:
                pourcentage_match_max = pourcentage_match
                id_best_match = i

    return id_best_match
            
        
    