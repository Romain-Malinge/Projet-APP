from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import cv2
import numpy as np
import track_gaze

def get_palette(num_classes):
    np.random.seed(42)
    palette = np.random.randint(0, 255, (num_classes, 3), dtype=np.uint8)
    return palette

def seg_former_on_video(video_path, db_path, max_seconds=15, skip_seconds=5, percentage=20, max_size=800):
    # Charger le modèle et le feature extractor
    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
    classes_to_keep = {
        0: "road",
        1: "sidewalk",
        2: "building",
        7: "traffic sign",
        11: "person",
        13: "car",
        17: "motorcycle"
    }
    allowed_ids = list(classes_to_keep.keys())
    gaze_ts, xs, ys = track_gaze.load_gaze_from_sqlite(db_path)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    max_frames = int(fps * max_seconds)
    palette = get_palette(model.config.num_labels)
        
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
        # Filtrer les classes non désirées
        mask = np.isin(seg, allowed_ids)
        seg = np.where(mask, seg, 0)  # 255 = ignore
        seg_img = Image.fromarray(seg.astype(np.uint8))
        seg_img = seg_img.resize(image.size, resample=Image.NEAREST)
        seg = np.array(seg_img)
        color_seg = palette[seg]
        image_np = np.array(image)
        overlay = (0.5 * image_np + 0.5 * color_seg).astype(np.uint8)

        # Convertir overlay en BGR pour affichage OpenCV
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imshow('Segformer Overlay', overlay_bgr)
        
        timestamp_s = int(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        gaze = track_gaze.find_gaze_for_frame(int(timestamp_s * 1e9), gaze_ts, xs, ys)
        if gaze is not None:
            gx, gy = gaze
            gx = int(gx)
            gy = int(gy)
            class_id = seg[int(gy * (target_size[1] / h)), int(gx * (target_size[0] / w))]
            class_name = classes_to_keep.get(class_id, "inconnu")
            print(f"Frame {frame_count}: Class Name: {class_name}")
        else:
            print(f"Frame {frame_count}: No gaze data available.")
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
    percentage = 100 # Pourcentage de frames à traiter
    max_size = 800 # Taille maximale pour la fenêtre
    seg_former_on_video(video_path, db_path, max_seconds, skip_seconds, percentage, max_size)