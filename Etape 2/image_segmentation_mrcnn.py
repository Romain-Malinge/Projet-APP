import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

# Mask R-CNN imports (assuming mrcnn library is installed)
# pip install mrcnn (Note: mrcnn requires specific versions of Keras, TensorFlow, and h5py)
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# COCO class names (80 categories)
COCO_CLASSES = [
    'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]


# --- Configuration for Inference ---
class InferenceConfig(Config):
    """
    Configuration for inference on the COCO dataset.
    """
    NAME = "coco_inference"
    NUM_CLASSES = 1 + 80  # COCO has 80 classes + background
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


# --- Helper Function for Visualization ---
def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    The mask is overlaid with a specific color."""
    
    # Check dimensions
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) # Convert to 3-channel if grayscale

    # Ensure mask is boolean
    mask = mask.astype(bool)
    
    # Create an overlay image with the color
    overlay = np.ones_like(image, dtype=np.uint8) * np.array(color, dtype=np.uint8)

    # Blend the original image and the color overlay where the mask is True
    # Formula: result = alpha * color + (1 - alpha) * original_image
    for c in range(image.shape[2]):
        image[:, :, c] = np.where(mask,
                                  image[:, :, c] * (1 - alpha) + overlay[:, :, c] * alpha,
                                  image[:, :, c])
    return image


def visualize_segmentation(image, results):
    """Draw bounding boxes and masks on image based on Mask R-CNN results."""
    
    # The results is a list of dictionaries, we only use the first item for single image inference
    r = results[0]
    masks = r['masks']
    rois = r['rois']
    class_ids = r['class_ids']
    scores = r['scores']
    
    N = masks.shape[-1]
    if N == 0:
        print("No instances to display.")
        return image
    
    # Convert image to RGB if not already
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 and image.shape[2] == 3 else image.copy()
    
    # Iterate over all detected instances
    for i in range(N):
        # Generate a random color for the mask and box
        color = [random.randint(0, 255) for _ in range(3)] 
        
        # Get mask
        mask = masks[:, :, i]
        
        # Apply mask overlay
        image_rgb = apply_mask(image_rgb, mask, color, alpha=0.5)

        # Get bounding box coordinates
        y1, x1, y2, x2 = rois[i]
        
        # Draw bounding box
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 2)
        
        # Get label and score
        label = COCO_CLASSES[class_ids[i]]
        score = scores[i]
        text = f"{label}: {score:.2f}"
        
        # Draw label text
        cv2.putText(image_rgb, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image_rgb


# --- Core Mask R-CNN Functions ---
def load_mask_rcnn_model():
    """Load pre-trained Mask R-CNN model."""
    config = InferenceConfig()
    
    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)

    # Download weights if needed
    if not os.path.exists(COCO_WEIGHTS_PATH):
        print("Downloading pre-trained COCO weights...")
        utils.download_trained_weights(COCO_WEIGHTS_PATH)

    # Load weights
    print(f"Loading weights from {COCO_WEIGHTS_PATH}")
    model.load_weights(COCO_WEIGHTS_PATH, by_name=True)
    return model


def segment_image(model, image_path, output_path=None):
    """Run Mask R-CNN on an image and show/save results."""
    
    # 1. Load image (using OpenCV, default BGR format)
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
        
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # Convert to RGB (Mask R-CNN expects RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 2. Detect objects
    print("Running detection on image...")
    # 
    results = model.detect([image_rgb], verbose=1)

    # 3. Visualize results
    print("Visualizing segmentation...")
    # The visualization function needs the original BGR image to work correctly with OpenCV color plotting 
    # but the image passed to model.detect was RGB. The visualization function handles this.
    result_img_rgb = visualize_segmentation(image, results)

    # 4. Display result
    plt.figure(figsize=(12, 12))
    plt.imshow(result_img_rgb)
    plt.title(f"Mask R-CNN Segmentation: {os.path.basename(image_path)}")
    plt.axis("off")
    plt.show()

    # 5. Save result if an output path is provided
    if output_path:
        # Convert the result back to BGR for saving with OpenCV
        cv2.imwrite(output_path, cv2.cvtColor(result_img_rgb, cv2.COLOR_RGB2BGR))
        print(f"Saved segmented output to {output_path}")


def main():
    """Main function to parse arguments and run the script."""
    image_path = "extracted_frame.jpg"
    output_path = "segmented_output.jpg"
    # Load the Mask R-CNN model
    model = load_mask_rcnn_model()
    
    # Run segmentation
    segment_image(model, image_path, output_path)


if __name__ == "__main__":
    main()