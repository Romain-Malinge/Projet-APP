from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import cv2
import numpy as np

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
# Paramètres vidéo
video_path = "./data/2025-11-20_15-30-11-a3a383b4/95cbe6dd_0.0-323.503.mp4" # Remplace par le chemin de ta vidéo
max_seconds = 15 # Durée maximale à traiter (en secondes)
skip_seconds = 5 # Nombre de secondes à sauter au début
percentage = 20 # Pourcentage de frames à traiter
max_size = 800
cap = cv2.VideoCapture(video_path)
frame_count = 0
fps = cap.get(cv2.CAP_PROP_FPS)
max_frames = int(fps * max_seconds)

def get_palette(num_classes):
	np.random.seed(42)
	palette = np.random.randint(0, 255, (num_classes, 3), dtype=np.uint8)
	return palette

palette = get_palette(model.config.num_labels)

print("=== Couleurs utilisées pour les classes sélectionnées ===")
for class_id, class_name in classes_to_keep.items():
    color = palette[class_id].tolist()
    print(f"{class_id:2d} | {class_name:15s} → RGB {color}")
    

fps = cap.get(cv2.CAP_PROP_FPS)
skip_frames = int(skip_seconds * fps)

cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
step = int(100 / percentage)
allowed_ids = list(classes_to_keep.keys())
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

	# Quitter si touche 'q' pressée
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
