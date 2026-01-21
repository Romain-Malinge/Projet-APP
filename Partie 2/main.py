from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import cv2
import numpy as np
import track_gaze

def get_palette(classes_to_keep):
    np.random.seed(42)
    # On crée une palette de taille = max_id + 1
    max_id = max(classes_to_keep.keys())
    palette = np.zeros((max_id + 1, 3), dtype=np.uint8)
    # On génère une couleur uniquement pour les IDs utilisés
    for cid in classes_to_keep.keys():
        palette[cid] = np.random.randint(0, 255, 3, dtype=np.uint8)
    return palette

def seg_former_on_video(video_path, db_path, max_seconds=15, skip_seconds=5, percentage=20, max_size=800):
    # Charger le modèle et le feature extractor
    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
    classes= {
        0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence', 
        5: 'pole', 6: 'traffic light', 7: 'traffic sign', 8: 'vegetation', 
        9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider', 13: 'car', 
        14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle'
    }
    regroupements = {
    0: 0, 1: 0, 2: 2, 3: 2, 4: 2, 5: 7, 6: 7, 8: 8, 9: 8, 10: 10, 11: 11, 12: 18,
    13: 13, 14: 13, 15: 13, 16: None, 17: 18, 18: 18}
    classes_to_keep = {0: 'route', 2: 'batiment', 7: 'panneau', 8: 'vegetation', 13: '4 roues', 18: '2 roues',
                       11: 'piéton', 10: 'ciel'}
    # Paramètres vidéo
    gaze_ts, xs, ys = track_gaze.load_gaze_from_sqlite(db_path)
    timestamp_0 = min(gaze_ts) // 1_000_000_000
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    max_frames = int(fps * max_seconds)
    palette = get_palette(classes_to_keep)
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    skip_frames = int(skip_seconds * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
    step = int(100 / percentage)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count >= max_frames:
            break
        frame_count += 1
        if frame_count % step != 0:
            continue
        h, w = frame.shape[:2]
        scale = max_size / max(h, w)
        target_size = (int(w * scale), int(h * scale))  # (largeur, hauteur)
        target_size = (1024, 1024)
        frame_resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        # Convertir BGR (OpenCV) en RGB (PIL)
        image = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        seg = np.argmax(logits.detach().cpu().numpy(), axis=1)[0]
        # Appliquer les regroupements
        seg_grouped = np.full_like(seg, fill_value=-1)
        for old_id, new_id in regroupements.items():
            if new_id is not None:
                seg_grouped[seg == old_id] = new_id
        seg_img = Image.fromarray(seg_grouped.astype(np.int16))
        seg_img = seg_img.resize(image.size, resample=Image.NEAREST)
        seg_filtered = np.array(seg_img)
        color_seg = np.zeros((*seg_filtered.shape, 3), dtype=np.uint8)
        for cid, _ in classes_to_keep.items():
            color_seg[seg_filtered == cid] = palette[cid]
        image_np = np.array(image)
        overlay = (0.5 * image_np + 0.5 * color_seg).astype(np.uint8)

        # Convertir overlay en BGR pour affichage OpenCV
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        timestamp_s = int(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0) + timestamp_0
        gaze = track_gaze.find_gaze_for_frame(int(timestamp_s * 1e9), gaze_ts, xs, ys)
        if gaze is not None:
            gx, gy = gaze
            gx = int(gx)
            gy = int(gy)
            class_id = seg_filtered[int(gy * (target_size[1] / h)), int(gx * (target_size[0] / w))]
            # Dessiner le point de regard
            gx_resized = int(gx * (target_size[0] / w))
            gy_resized = int(gy * (target_size[1] / h))
            cv2.circle(overlay_bgr, (gx_resized, gy_resized), 6, (0, 0, 255), -1)
            class_name = classes_to_keep.get(class_id, "inconnu")
            print(f"Frame {frame_count}: Class Name: {class_name}")
        else:
            print(f"Frame {frame_count}: No gaze data available.")
        cv2.imshow('Segformer Overlay', overlay_bgr)
        # Quitter si touche 'q' pressée
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    db_path = "./data/database1.sqlite"
    video_path = "./data/2025-11-20_15-30-11-a3a383b4/95cbe6dd_0.0-323.503.mp4" # Remplace par le chemin de ta vidéo
    max_seconds = 15 # Durée maximale à traiter (en secondes)
    skip_seconds = 5 # Nombre de secondes à sauter au début
    percentage = 20 # Pourcentage de frames à traiter
    max_size = 800 # Taille maximale pour la fenêtre
    seg_former_on_video(video_path, db_path, max_seconds, skip_seconds, percentage, max_size)