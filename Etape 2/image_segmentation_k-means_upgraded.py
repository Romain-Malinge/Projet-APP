import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

def segment_urban_scene_lxy(image_path, min_k=2, max_k=10, quick_search_width=500):
    """
    Segments an urban image using K-means with L*a*b* color features 
    combined with normalized X and Y spatial coordinates (L*a*b*X Y).
    
    Args:
        image_path (str): Path to the image file.
        min_k (int): Minimum number of clusters to test.
        max_k (int): Maximum number of clusters to test.
        quick_search_width (int): Width to resize image for the k-search.
    """
    
    # 1. Load and Preprocess Image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: Could not load image at {image_path}")
        return

    original_shape = img_bgr.shape
    H, W, _ = original_shape

    # 2. Prepare Data for K-Search (Downsampling)
    aspect_ratio = H / W
    new_height = int(quick_search_width * aspect_ratio)
    
    # --- Prepare Data for K-Means (L*a*b*X Y Features) ---
    
    def prepare_features(input_img, target_W, target_H):
        """Generates the L*a*b*X Y feature matrix for K-means."""
        
        # Resize image for fast search or use original size for final segmentation
        resized_img = cv2.resize(input_img, (target_W, target_H))
        
        # Convert to L*a*b* color space
        img_lab = cv2.cvtColor(resized_img, cv2.COLOR_BGR2LAB)
        
        # Reshape L*a*b* data: (N_pixels, 3)
        data_lab = img_lab.reshape((-1, 3)).astype(np.float32)

        # Generate X and Y spatial coordinates
        y_coords, x_coords = np.mgrid[0:target_H, 0:target_W]
        
        # Normalize X and Y to range [0, 1]
        x_norm = x_coords / target_W
        y_norm = y_coords / target_H
        
        # Reshape X and Y: (N_pixels, 1)
        data_x = x_norm.reshape((-1, 1)).astype(np.float32)
        data_y = y_norm.reshape((-1, 1)).astype(np.float32)
        
        # Combine all 5 features: [L, a, b, X, Y]
        feature_matrix = np.hstack((data_lab, data_x, data_y))
        
        # IMPORTANT: Weighting the spatial vs. color features.
        # Since X, Y are [0, 1], and L*a*b* are [0, 255] or [0, 127], 
        # the color features would dominate. We normalize L*a*b* too, 
        # then re-scale spatial coordinates slightly to bias segmentation.
        
        # Scale L*a*b* features to [0, 1]
        scaler = MinMaxScaler()
        feature_matrix[:, :3] = scaler.fit_transform(feature_matrix[:, :3])
        
        # Apply a weight to spatial coordinates (e.g., 0.5) to balance 
        # their influence against color features.
        spatial_weight = 0.5 
        feature_matrix[:, 3:] *= spatial_weight
        
        return feature_matrix, resized_img.shape


    # Prepare features for the fast search (downsampled image)
    data_small, shape_small = prepare_features(img_bgr, quick_search_width, new_height)
    
    print(f"--- Searching for optimal K between {min_k} and {max_k} ---")
    print(f"Search performed on data with 5 features and {data_small.shape[0]} samples.")

    best_k = min_k
    best_score = -1
    scores = []

    # 3. Iterate to find the best K (Silhouette Score)
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data_small)
        
        # Calculate Silhouette Score (computationally intensive)
        score = silhouette_score(data_small, labels)
        scores.append(score)
        
        print(f"k={k}: Silhouette Score = {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_k = k

    print(f"\n>>> Best K found: {best_k} (Score: {best_score:.4f})")
    
    # 4. Apply Best K to Full Resolution Image
    print("--- Applying segmentation to full resolution image ---")
    
    # Prepare features for the final segmentation (full size)
    data_full, _ = prepare_features(img_bgr, W, H)
    
    # Fit final model
    final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    final_labels = final_kmeans.fit_predict(data_full)
    
    # --- GENERATE BRIGHT COLOR PALETTE ---
    # Generate k colors evenly spaced around the hue wheel
    hues = np.linspace(0, 179, best_k, endpoint=False, dtype=np.uint8)
    hsv_palette = np.zeros((best_k, 1, 3), dtype=np.uint8)
    hsv_palette[:, 0, 0] = hues
    hsv_palette[:, 0, 1] = 255 # Max Saturation
    hsv_palette[:, 0, 2] = 255 # Max Value (Brightness)

    # Convert HSV palette to BGR (OpenCV format)
    bgr_bright_palette = cv2.cvtColor(hsv_palette, cv2.COLOR_HSV2BGR)
    bgr_bright_palette = bgr_bright_palette.reshape((best_k, 3))
    
    # Map the labels to the bright BGR colors
    segmented_data_bgr = bgr_bright_palette[final_labels]
    # Reshape back to original image dimensions (in BGR format)
    segmented_img_bgr = segmented_data_bgr.reshape(original_shape)

    # Convert BGR segmented image to RGB for matplotlib display
    segmented_img_rgb = cv2.cvtColor(segmented_img_bgr, cv2.COLOR_BGR2RGB)
    
    # 5. Visualization
    plt.figure(figsize=(14, 6))
    
    # Plot Original (Convert BGR to RGB for correct display)
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.title("Original Urban Scene")
    plt.axis('off')

    # Plot Scores
    plt.subplot(1, 3, 2)
    plt.plot(range(min_k, max_k + 1), scores, marker='o', color='purple')
    plt.title("Silhouette Scores vs K")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Score")
    plt.grid(True, linestyle='--')

    # Plot Segmented (L*a*b*X Y)
    plt.subplot(1, 3, 3)
    plt.imshow(segmented_img_rgb)
    plt.title(f"Segmented (k={best_k}) - L*a*b*X Y")
    plt.axis('off')

    plt.tight_layout()
    print("Displaying results of spatially-aware K-means...")
    plt.show()

# --- Usage Example ---
# Replace 'path/to/your/urban_image.jpg' with a real file path
segment_urban_scene_lxy('extracted_frame.jpg', min_k=5, max_k=20)