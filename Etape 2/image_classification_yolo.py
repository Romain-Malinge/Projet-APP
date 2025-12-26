import cv2
from ultralytics import YOLO

def yolo_detect_and_classify(image_path, model_name='yolov8n.pt', confidence_threshold=0.5):
    """
    Uses a pretrained YOLOv8 model to detect objects in an image and
    print the detected objects and their classifications.
    
    Args:
        image_path (str): Path to the input image file.
        model_name (str): The YOLO model version (e.g., 'yolov8n.pt' for nano).
        confidence_threshold (float): Minimum confidence to show a detection.
    """
    print(f"--- Loading YOLO Model: {model_name} ---")
    
    try:
        # Load the pretrained YOLO model (downloads if not present)
        model = YOLO(model_name)
    except Exception as e:
        print(f"Error loading model. Make sure you have internet access and the model name is correct: {e}")
        return

    # 1. Run inference on the image
    # The 'results' object contains the detection information (boxes, classes, confidences)
    print(f"--- Running Inference on {image_path} ---")
    results = model(image_path, conf=confidence_threshold)

    # 2. Process and Display Results
    
    if not results or not results[0].boxes:
        print("No objects detected above the confidence threshold.")
        return

    # Get the results for the first image (since we only pass one)
    detection = results[0]
    
    print("\n--- Detected Objects ---")
    
    # Loop through all detected boxes
    for box in detection.boxes:
        confidence = box.conf.item() * 100
        class_id = box.cls.item()
        
        # Get the human-readable class name from the model's names dictionary
        class_name = model.names[int(class_id)]
        
        # Get the bounding box coordinates (normalized or pixel)
        # Using xyxy for pixel coordinates: [x_min, y_min, x_max, y_max]
        coords = box.xyxy[0].cpu().numpy().astype(int)
        
        print(f"  - Class: **{class_name}**")
        print(f"    Confidence: {confidence:.2f}%")
        print(f"    Bounding Box (xyxy): {coords}")

    # 3. Optional: Save the image with bounding boxes
    try:
        # Plot the detections onto the image
        annotated_image_path = "yolo_output_annotated.jpg"
        detection.save(annotated_image_path)
        print(f"\n--- Output saved to {annotated_image_path} ---")
    except Exception as e:
        print(f"Could not save annotated image: {e}")


# --- Usage Example ---
# IMPORTANT: Replace 'my_urban_image.jpg' with the path to your actual image file.
yolo_detect_and_classify('extracted_frame.jpg', model_name='yolov8s.pt') # 's' is small