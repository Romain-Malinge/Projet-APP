from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import cv2
import numpy as np
import track_gaze
from chronique_temporelle import ChroniqueTemporelle

# classes= {
#         0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence', 
#         5: 'pole', 6: 'traffic light', 7: 'traffic sign', 8: 'vegetation', 
#         9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider', 13: 'car', 
#         14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle'
#     }
def get_palette(classes_to_keep):
    np.random.seed(42)
    # On crée une palette de taille = max_id + 1
    max_id = max(classes_to_keep.keys())
    palette = np.zeros((max_id + 1, 3), dtype=np.uint8)
    # On génère une couleur uniquement pour les IDs utilisés
    for cid in classes_to_keep.keys():
        palette[cid] = np.random.randint(0, 255, 3, dtype=np.uint8)
    return palette

def charger_modele():
    feature_extractor = SegformerFeatureExtractor.from_pretrained(
        "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
    )
    return feature_extractor, model

def charger_gaze(db_path):
    gaze_ts, xs, ys = track_gaze.load_gaze_from_sqlite(db_path)
    gaze_ts = gaze_ts - gaze_ts[0] # recentrer les gazes
    return gaze_ts, xs, ys

def segmenter_frame(frame_resized, feature_extractor, model):
    image = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
    inputs = feature_extractor(images=image, return_tensors="pt")
    logits = model(**inputs).logits
    seg = np.argmax(logits.detach().cpu().numpy(), axis=1)[0]
    return seg, image

def regrouper_classes(seg, regroupements):
    seg_grouped = np.full_like(seg, -1)
    for old_id, new_id in regroupements.items():
        if new_id is not None:
            seg_grouped[seg == old_id] = new_id
    return seg_grouped

def coloriser_segmentation(seg_grouped, image, classes_to_keep, palette):
    seg_img = Image.fromarray(seg_grouped.astype(np.int16))
    seg_img = seg_img.resize(image.size, resample=Image.NEAREST)
    seg_filtered = np.array(seg_img)
    color_seg = np.zeros((*seg_filtered.shape, 3), dtype=np.uint8)
    for cid in classes_to_keep:
        color_seg[seg_filtered == cid] = palette[cid]

    return seg_filtered, color_seg

def traiter_gaze(gaze, seg_filtered, w, h, target_size, classes_to_keep):
    if gaze is None:
        return None, None, None

    gx, gy = gaze
    x_seg = int(gx * (target_size[0] / w))
    y_seg = int(gy * (target_size[1] / h))
    class_id = seg_filtered[y_seg, x_seg]
    class_name = classes_to_keep.get(class_id, "inconnu")
    return x_seg, y_seg, class_name

def seg_former_on_video(video_path, db_path, max_seconds=15, skip_seconds=5, percentage=20, max_size=800):

    feature_extractor, model = charger_modele()
    gaze_ts, xs, ys = charger_gaze(db_path)
    regroupements = {
    0: 0, 1: 0, 2: 2, 3: 2, 4: 2, 5: 7, 6: 7, 8: 8, 9: 8, 10: None, 11: 11, 12: 18,
    13: 13, 14: 13, 15: 13, 16: None, 17: 18, 18: 18}
    classes_to_keep = {0: 'route', 2: 'batiment', 7: 'panneau', 8: 'vegetation', 13: '4 roues', 18: '2 roues',
                       11: 'piéton'} #, 10: 'ciel'
    palette = get_palette(classes_to_keep)
    chronique = ChroniqueTemporelle(classes_to_keep.values())
    # Paramètres vidéo
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    max_frames = int(fps * max_seconds)
    skip_frames = int(skip_seconds * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
    step = int(100 / percentage)
    frame_count = 0
    target_size = (1024, 1024)
    # Enregistrer la vidéo
    output_path = "video_segmentation2.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, target_size)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break

        frame_count += 1
        if frame_count % step != 0:
            continue

        h, w = frame.shape[:2]
        scale = max_size / max(h, w)
        target_size = (int(w * scale), int(h * scale))  # (largeur, hauteur)
        frame_resized = cv2.resize(frame, target_size)

        
        seg, image = segmenter_frame(frame_resized, feature_extractor, model)
        seg_grouped = regrouper_classes(seg, regroupements)
        seg_filtered, color_seg = coloriser_segmentation(seg_grouped, image, classes_to_keep, palette)

        overlay = (0.5 * np.array(image) + 0.5 * color_seg).astype(np.uint8)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        timestamp_s = int(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        gaze = track_gaze.find_gaze_for_frame(int(timestamp_s * 1e9), gaze_ts, xs, ys)

        x_seg, y_seg, class_name = traiter_gaze(gaze, seg_filtered, w, h, target_size, classes_to_keep)
        cv2.circle(overlay_bgr, (x_seg, y_seg), 6, (0, 0, 255), -1)
        if class_name and class_name != "inconnu":
            print(f"Frame {frame_count}: Class Name: {class_name}")
            chronique.ajouter_frame(class_name, frame_count)

        cv2.imshow("Segformer Overlay", overlay_bgr)
        if overlay_bgr.shape[1] != 1024 or overlay_bgr.shape[0] != 1024:
            overlay_bgr = cv2.resize(overlay_bgr, (1024, 1024))
        writer.write(overlay_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    return chronique

    
if __name__ == "__main__":
    # 15min pour vidéo 1, 5min->305s avec skip=15s
    db_path = "./data/database1.sqlite" # Remplacer par le chemin de la base de données
    video_path = "./data/2025-11-20_15-30-11-a3a383b4/95cbe6dd_0.0-323.503.mp4" # Remplacer par le chemin de la vidéo
    max_seconds = 10 # Durée maximale à traiter (en secondes)
    skip_seconds = 15 # Nombre de secondes à sauter au début
    percentage = 10 # Pourcentage de frames à traiter
    max_size = 800 # Taille maximale pour la fenêtre
    chronique_temporelle = seg_former_on_video(video_path, db_path, max_seconds, skip_seconds, percentage, max_size)
    chronique_temporelle.afficher()