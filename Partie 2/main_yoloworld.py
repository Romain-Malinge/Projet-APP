import cv2
from ultralytics import YOLOWorld

# Charger le modèle
model = YOLOWorld("yolov8m-world.pt") # ou "yolo8s-world.pt" (plus rapide/moins précis)
model.set_classes(["motorcycle", "car", "person", "road", "sidewalk", "building", "sign"])

# Ouverture vidéo (se mettre dans Partie 2 pour lancer)
video_path = "./data/2025-11-20_15-30-11-a3a383b4/95cbe6dd_0.0-323.503.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Erreur : impossible d'ouvrir la vidéo")
    exit()

# Durée maximale à traiter (en secondes)
max_seconds = 10

# Récupérer le nombre d'images par seconde
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0
max_frames = int(fps * max_seconds)

ret, frame = cap.read()
if not ret:
    print("Impossible de lire la vidéo")
    exit()

# Dimensions de la vidéo
h, w = frame.shape[:2]
max_size = 800
scale = max_size / max(h, w)
new_w = int(w * scale)
new_h = int(h * scale)
# Créer la fenêtre redimensionnée
cv2.namedWindow("YOLOWorld - Test sur X secondes", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOWorld - Test sur X secondes", new_w, new_h)

# Revenir au début de la vidéo
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


while True:
    ret, frame = cap.read()
    if not ret:
        break  # Fin de la vidéo

    # Arrêter après X secondes
    if frame_count >= max_frames:
        break

    frame_count += 1

    # Conversion BGR → RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Reshape -> détection + rapide
    frame_small = cv2.resize(frame_rgb, (640, 360))
    # Prédiction YOLOWorld
    results = model.predict(frame_small, verbose=False)
    sx = frame.shape[1] / 640
    sy = frame.shape[0] / 360
    boxes = results[0].boxes

    # Dessiner les boîtes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1 = int(x1 * sx)
        y1 = int(y1 * sy)
        x2 = int(x2 * sx)
        y2 = int(y2 * sy)

        label = results[0].names[int(box.cls[0])]
        score = float(box.conf[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {score:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    # Affichage
    cv2.imshow("YOLOWorld - Test sur X secondes", frame)

    # Quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
