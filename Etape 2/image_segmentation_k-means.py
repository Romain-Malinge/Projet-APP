import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def segment_image_auto_k_bright(image_path, min_k=2, max_k=10, quick_search_width=100):
    """
    Segments an image using K-means, automatically selecting the best k,
    and displays the result using distinct, bright colors.
    """
    
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # Convert from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_shape = img_rgb.shape

    # 2. Preprocessing for K-Search (Downsampling)
    aspect_ratio = original_shape[0] / original_shape[1]
    new_height = int(quick_search_width * aspect_ratio)
    img_small = cv2.resize(img_rgb, (quick_search_width, new_height))
    data_small = img_small.reshape((-1, 3))

    print(f"--- Searching for optimal K between {min_k} and {max_k} ---")

    best_k = min_k
    best_score = -1
    scores = []

    # 3. Iterate to find the best K (calculating Silhouette Score)
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data_small)
        score = silhouette_score(data_small, labels)
        scores.append(score)
        print(f"k={k}: Silhouette Score = {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_k = k

    print(f"\n>>> Best K found: {best_k} (Score: {best_score:.4f})")
    
    # 4. Apply Best K to Full Resolution Image
    print("--- Applying segmentation to full resolution image ---")
    data_full = img_rgb.reshape((-1, 3))
    
    # Fit final model
    final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    final_labels = final_kmeans.fit_predict(data_full)
    
    # Get natural centers for comparison
    natural_centers = np.uint8(final_kmeans.cluster_centers_)

    # --- GENERATE BRIGHT COLOR PALETTE ---
    # We generate k colors evenly spaced around the hue wheel in HSV space.
    # OpenCV HSV ranges: H [0-179], S [0-255], V [0-255]
    
    # 1. Generate distinct Hues
    hues = np.linspace(0, 179, best_k, endpoint=False, dtype=np.uint8)
    
    # 2. Create HSV palette (Shape needs to be (N, 1, 3) for cvtColor function)
    hsv_palette = np.zeros((best_k, 1, 3), dtype=np.uint8)
    hsv_palette[:, 0, 0] = hues       # Hue varies
    hsv_palette[:, 0, 1] = 255        # Saturation max
    hsv_palette[:, 0, 2] = 255        # Value (brightness) max

    # 3. Convert HSV palette back to RGB for display
    rgb_bright_palette = cv2.cvtColor(hsv_palette, cv2.COLOR_HSV2RGB)
    
    # Reshape back to a simple list of k colors: (k, 3)
    rgb_bright_palette = rgb_bright_palette.reshape((best_k, 3))

    # 5. Reconstruct Images
    # A) Natural Segmentation (using average colors)
    segmented_data_natural = natural_centers[final_labels]
    segmented_img_natural = segmented_data_natural.reshape(original_shape)

    # B) Bright Segmentation (using custom palette)
    # Map the labels (0, 1, 2...) to index our new bright palette
    segmented_data_bright = rgb_bright_palette[final_labels]
    # Reshape back to original image dimensions
    segmented_img_bright = segmented_data_bright.reshape(original_shape)


    # 6. Visualization (Updated for 4 panels)
    plt.figure(figsize=(16, 6))
    
    # Plot Original
    plt.subplot(1, 4, 1)
    plt.imshow(img_rgb)
    plt.title("Original")
    plt.axis('off')

    # Plot Scores
    plt.subplot(1, 4, 2)
    plt.plot(range(min_k, max_k + 1), scores, marker='o', color='purple')
    plt.title("Optimal K Search")
    plt.xlabel("k")
    plt.ylabel("Silhouette Score")
    plt.grid(True, linestyle='--')

    # Plot Natural Segmented
    plt.subplot(1, 4, 3)
    plt.imshow(segmented_img_natural)
    plt.title(f"Natural Colors (k={best_k})")
    plt.axis('off')
    
    # Plot Bright Segmented
    plt.subplot(1, 4, 4)
    plt.imshow(segmented_img_bright)
    plt.title(f"Bright Segmentation (k={best_k})")
    plt.axis('off')

    plt.tight_layout()
    print("Displaying results...")
    plt.show()

# --- Usage Example ---
# Replace 'path/to/your/image.jpg' with a real file path
segment_image_auto_k_bright('extracted_frame.jpg', min_k=8, max_k=16)