from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import cv2
import numpy as np
import track_gaze
from chronique_temporelle import ChroniqueTemporelle
cityscapes_colors = {
        0: (128, 64, 128), 1: (244, 35, 232), 2: (0, 80, 100),
        3: (70, 70, 70), 4: (190, 153, 153), 5: (153, 153, 153),
        6: (250, 170, 30), 7: (220, 220, 0), 8: (107, 142, 35),
        9: (152, 251, 152), 10: (70, 130, 180), 11: (220, 20, 60),
        12: (255, 0, 0), 13: (0, 0, 142), 14: (0, 0, 70),
        15: (0, 60, 100), 16: (0, 80, 100),
        17: (0, 0, 230), 18: (119, 11, 32)
    }
def segformer_on_video_frame(video_path, frame_number=0, output_path="segformer_output.png", db_path=None, gaze=None, max_size=800):
    """
    Extrait une frame d'une vidéo, applique la segmentation, sauvegarde l'overlay colorisé.
    """
    gaze_ts, xs, ys = charger_gaze(db_path)
    feature_extractor, model = charger_modele()
    regroupements = {
        0: 0, 1: 0, 2: 2, 3: 2, 4: 2, 5: 7, 6: 7, 7: 7, 8: 8, 9: 8, 10: 10, 11: 11, 12: 18,
        13: 13, 14: 13, 15: 13, 16: None, 17: 18, 18: 18}
    classes_to_keep = {0: 'route', 2: 'batiment', 7: 'panneau', 8: 'vegetation', 13: '4 roues', 18: '2 roues', 11: 'piéton', 10: 'ciel'}
    palette = cityscapes_colors

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_number < 0 or frame_number >= total_frames:
        raise ValueError(f"Numéro de frame invalide : {frame_number} (max = {total_frames-1})")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Impossible de lire la frame {frame_number} de {video_path}")
    h, w = frame.shape[:2]
    scale = max_size / max(h, w)
    target_size = (int(w * scale), int(h * scale))
    frame_resized = cv2.resize(frame, target_size)

    seg, image = segmenter_frame(frame_resized, feature_extractor, model)
    seg_grouped = regrouper_classes(seg, regroupements)
    seg_filtered, color_seg = coloriser_segmentation(seg_grouped, image, classes_to_keep, palette)

    overlay = (0.3 * np.array(image) + 0.7 * color_seg).astype(np.uint8)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    gaze = track_gaze.find_gaze_for_frame(frame_number*30/1e9, gaze_ts, xs, ys)
    class_name = None
    if gaze is not None:
        gx, gy = gaze
        x_seg = int(gx * (target_size[0] / w))
        y_seg = int(gy * (target_size[1] / h))
        class_id = seg_filtered[y_seg, x_seg]
        class_name = classes_to_keep.get(class_id, "inconnu")
        cv2.circle(overlay_bgr, (x_seg, y_seg), 6, (0, 0, 255), -1)
        print(f"Classe au point de regard : {class_name}")

    cv2.imwrite(output_path, overlay_bgr)
    print(f"Image sauvegardée sous : {output_path}")
    return overlay_bgr, class_name

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
def charger_gaze(db_path):
    gaze_ts, xs, ys = track_gaze.load_gaze_from_sqlite(db_path)
    gaze_ts = gaze_ts - gaze_ts[0] # recentrer les gazes
    return gaze_ts, xs, ys

if __name__ == "__main__":
    # Exemple d'utilisation pour une frame d'une vidéo
    db_path = "./data/database1.sqlite"
    video_path = "./data/2025-11-20_15-30-11-a3a383b4/95cbe6dd_0.0-323.503.mp4"  # À adapter
    frame_number = 7870  # Numéro de la frame à traiter
    output_path = "segformer_output.png"
    segformer_on_video_frame(video_path, frame_number, output_path,db_path = db_path,)