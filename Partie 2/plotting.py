import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.patches as mpatches

def plot_segmentation_non_blocking(ax, original, segmentation, frame_num):
    """Version non-blocante de plot_segmentation"""
    cityscapes_colors = {
        0: (128, 64, 128), 1: (244, 35, 232), 2: (70, 70, 70),
        3: (102, 102, 156), 4: (190, 153, 153), 5: (153, 153, 153),
        6: (250, 170, 30), 7: (220, 220, 0), 8: (107, 142, 35),
        9: (152, 251, 152), 10: (70, 130, 180), 11: (220, 20, 60),
        12: (255, 0, 0), 13: (0, 0, 142), 14: (0, 0, 70),
        15: (0, 60, 100), 16: (0, 80, 100), 17: (0, 0, 230), 18: (119, 11, 32)
    }
    
    cityscapes_labels = {
        0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence', 
        5: 'pole', 6: 'traffic light', 7: 'traffic sign', 8: 'vegetation', 
        9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider', 13: 'car', 
        14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle'
    }
    
    # Clear previous plot
    ax.clear()
    
    # Segmentation RGB overlay
    segmentation_rgb = np.zeros((*segmentation.shape, 3), dtype=np.uint8)
    for class_id, color in cityscapes_colors.items():
        segmentation_rgb[segmentation == class_id] = color
    
    # Affichage avec overlay
    ax.imshow(original)
    ax.imshow(segmentation_rgb, alpha=0.5)
    
    ax.set_title(f'Semantic Segmentation - Frame {frame_num}')
    ax.axis('off')
    
    # Legend dynamique (seulement classes présentes)
    patches = []
    unique_classes = np.unique(segmentation)
    for class_id in unique_classes:
        if class_id in cityscapes_labels:
            color_tuple = tuple(c / 255.0 for c in cityscapes_colors[class_id])
            patches.append(mpatches.Patch(color=color_tuple, label=f'{cityscapes_labels[class_id]}'))
    
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.draw()
    plt.pause(0.01)  # Pause très courte pour rafraîchir

def show_segmentation_opencv(original_rgb, segmentation, gaze_x, gaze_y, alpha=0.5):
    """
    Affichage OpenCV de la segmentation sémantique en overlay
    """
    cityscapes_colors = {
        0: (128, 64, 128), 1: (244, 35, 232), 2: (70, 70, 70),
        3: (102, 102, 156), 4: (190, 153, 153), 5: (153, 153, 153),
        6: (250, 170, 30), 7: (220, 220, 0), 8: (107, 142, 35),
        9: (152, 251, 152), 10: (70, 130, 180), 11: (220, 20, 60),
        12: (255, 0, 0), 13: (0, 0, 142), 14: (0, 0, 70),
        15: (0, 60, 100), 16: (0, 80, 100),
        17: (0, 0, 230), 18: (119, 11, 32)
    }

    # Création masque couleur
    color_mask = np.zeros_like(original_rgb, dtype=np.uint8)
    for class_id, color in cityscapes_colors.items():
        color_mask[segmentation == class_id] = color

    # Cercle de couleur rouge aux coordonnées gaze_x, gaze_y
    cv2.circle(original_rgb, (int(gaze_x), int(gaze_y)), 20, (0, 0, 255), -1)

    # Overlay
    overlay = cv2.addWeighted(original_rgb, 1 - alpha, color_mask, alpha, 0)

    # OpenCV attend BGR
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

    cv2.imshow("Semantic Segmentation", overlay_bgr)

    cv2.waitKey(1)

