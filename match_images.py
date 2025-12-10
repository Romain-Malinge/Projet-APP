import cv2
import numpy as np


def apply_orb(image):
    # Image en niveaux de gris
    orb = cv2.ORB.create(nfeatures = 3000)
    keypoints,descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def match_keypoints(descriptors_src, descriptors_dest):
    """
    Calcule les correspondances entre deux descripteurs (src -> dest)
    descriptors_src : descripteur de l'image gauche
    descriptors_dest : descripteur de l'image droite
    """
    # match selon la distance minimale entre descripteurs
    bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=False) # True -> contrainte de symétrie
    matches = bf.knnMatch(descriptors_src,descriptors_dest,k=2)
    good_matches = []
    for m,n in matches:
        if m.distance < 0.8 * n.distance: # Ratio test 
            good_matches.append(m)
    return good_matches


def match_and_display(img1, img2, max_matches=50):
    """
    Détecte et affiche les correspondances orb entre deux images.
    img1 : première image.
    img2 : deuxième image.
    max_matches : nombre maximum de correspondances à afficher.
    """
    # Charger les images en niveaux de gris
    kp1,d1 = apply_orb(img1)
    kp2,d2 = apply_orb(img2)
    matches = match_keypoints(d1,d2)
    h, w = img1.shape[:2]
    max_size = 800  # taille max pour l’affichage
    scale = max_size/max(h, w)
    # Dessiner les correspondances
    matched_img = cv2.drawMatches(
        img1, kp1, img2, kp2, matches[:min(max_matches,len(matches))], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    img_display = cv2.resize(matched_img, (int(w * scale), int(h * scale)))
    cv2.imshow("orb Matches", img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def match_homography(img_source, img_dest, kp_desc_src, kp_desc_dest, show=True):
    """
    Calcule l'homographie de img_source à img_dest et applique la transformation.
    (img_dest = H*img_source) Renvoie le pourcentage de points gardés.
    img_source : image source
    img_dest : image destination
    kp_desc_src : couple (keypoints,descripteurs) de l'image source 
    kp_desc_dest : couple (keypoints,descripteurs) de l'image destination
    show : affiche le résultat si True.
    """
    kp1, des1 = kp_desc_src
    kp2, des2 = kp_desc_dest
    matches = match_keypoints(des1, des2)
    if len(matches) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,6.0)
        if H is not None :
            if show:
                h, w = img_dest.shape[:2]
                img_warped = cv2.warpPerspective(img_source, H, (w, h))
                match_and_display(img_warped,img_dest, max_matches=50)
            return 100*mask.sum()/len(src_pts), H
        else:
            print("Attention : homographie impossible à calculer")
            return None, None
    else:
        print("Attention : pas assez de matches pour l'homographie")
        return None, None
    
def match_all(all_posters,frame_video,kp_desc_all_posters):
    """
    Fait les matches de la frame de la vidéo avec tous les posters
    Renvoie le poster avec le plus de matches (-1 si aucun ne convient), avec l'homographie associée.
    all_posters : liste de toutes les affiches
    frame_video : frame de la vidéo avec distorsion corrigée
    kp_desc_frame : (keypoints,descripteurs) de la frame
    kp_desc_all_posters : liste des (keypoints,descripteurs) de tous les posters
    """
    id_best_match = -1
    pourcentage_match_max = 0.0
    H_max = None
    frame_video = cv2.cvtColor(frame_video,cv2.COLOR_BGR2GRAY)
    kp_frame,desc_frame = apply_orb(frame_video)
    for i in range(len(all_posters)):
        kp_i,desc_i = kp_desc_all_posters[i]
        pourcentage_match, H = match_homography(frame_video,all_posters[i],(kp_frame,desc_frame),(kp_i,desc_i),False)
        if pourcentage_match is not None:
            #_, descriptors_warped = apply_orb(video_warped)
            #matches = match_keypoints(descriptors_warped,desc_i)
            #pourcentage_match = 100*len(matches)/len(desc_i)
            # print(pourcentage_match)
            if pourcentage_match > pourcentage_match_max and pourcentage_match >= 20:
                pourcentage_match_max = pourcentage_match
                id_best_match = i
                H_max = H
    return id_best_match, H_max
            
        
    