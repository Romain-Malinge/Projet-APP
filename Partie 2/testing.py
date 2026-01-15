import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# load Mask2Former fine-tuned on Cityscapes semantic segmentation
processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")

url = "https://images.pexels.com/photos/139303/pexels-photo-139303.jpeg?cs=srgb&dl=pexels-joshsorenson-139303.jpg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# model predicts class_queries_logits of shape `(batch_size, num_queries)`
# and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
class_queries_logits = outputs.class_queries_logits
masks_queries_logits = outputs.masks_queries_logits

# you can pass them to processor for postprocessing
predicted_semantic_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
# we refer to the demo notebooks for visualization (see "Resources" section in the Mask2Former docs)

# Assuming predicted_semantic_map is a tensor, convert to numpy array
predicted_semantic_map_np = predicted_semantic_map.cpu().numpy()

# Define Cityscapes colormap and labels
# These are standard colors for Cityscapes dataset classes
cityscapes_colors = {
    0: (128, 64, 128),   # road
    1: (244, 35, 232),   # sidewalk
    2: (70, 70, 70),     # building
    3: (102, 102, 156),  # wall
    4: (190, 153, 153),  # fence
    5: (153, 153, 153),  # pole
    6: (250, 170, 30),   # traffic light
    7: (220, 220, 0),    # traffic sign
    8: (107, 142, 35),   # vegetation
    9: (152, 251, 152),  # terrain
    10: (70, 130, 180),  # sky
    11: (220, 20, 60),    # person
    12: (255, 0, 0),      # rider
    13: (0, 0, 142),      # car
    14: (0, 0, 70),       # truck
    15: (0, 60, 100),     # bus
    16: (0, 80, 100),     # train
    17: (0, 0, 230),      # motorcycle
    18: (119, 11, 32),    # bicycle
    # Add more classes if your model predicts them, or map to 'unlabeled' or similar
}

cityscapes_labels = {
    0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence', 
    5: 'pole', 6: 'traffic light', 7: 'traffic sign', 8: 'vegetation', 
    9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider', 13: 'car', 
    14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle'
}

# Create an empty RGB image for the segmentation overlay
segmentation_rgb = np.zeros((*predicted_semantic_map_np.shape, 3), dtype=np.uint8)

# Populate the RGB image with colors based on the semantic map
for class_id, color in cityscapes_colors.items():
    segmentation_rgb[predicted_semantic_map_np == class_id] = color

plt.figure(figsize=(15, 10))

# Display the original image
plt.imshow(image)

# Overlay the segmentation map with transparency
# The alpha value can be adjusted (e.g., 0.5 for 50% transparency)
plt.imshow(segmentation_rgb, alpha=0.7)

plt.axis('off') # Hide axes ticks and labels
plt.title('Semantic Segmentation Overlay with Legend')

# Create a legend
patches = []
unique_classes_in_map = np.unique(predicted_semantic_map_np)
for class_id in unique_classes_in_map:
    if class_id in cityscapes_labels and class_id in cityscapes_colors:
        label = cityscapes_labels[class_id]
        color_tuple = tuple(c / 255.0 for c in cityscapes_colors[class_id]) # Normalize to 0-1 for matplotlib
        patches.append(mpatches.Patch(color=color_tuple, label=label))

# Place the legend outside the plot area for better visibility
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout() # Adjust layout to prevent overlapping elements
plt.show()