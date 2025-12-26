import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- I. FEATURE EXTRACTION FUNCTION ---

def extract_features_from_segments(img, k_clusters=5):
    """
    Performs K-means segmentation and extracts features for classification.

    The features for each cluster (segment) are:
    1. Average L*a*b* color (3 features)
    2. Normalized Area/Size (1 feature)
    3. Normalized Centroid Y (vertical position) (1 feature)
    
    Total features per segment: 5
    """
    H, W, _ = img.shape
    total_pixels = H * W
    
    # 1. Preprocessing: Reshape and convert to L*a*b* for K-means
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    data_lab = img_lab.reshape((-1, 3)).astype(np.float32)

    # 2. Run K-means Segmentation
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data_lab)
    
    features = []
    
    # 3. Extract Features for Each Segment (Cluster)
    for i in range(k_clusters):
        # Find all pixels belonging to the current cluster 'i'
        mask_flat = (labels == i)
        
        # A. Average Color (L*a*b*) - 3 features
        segment_pixels = data_lab[mask_flat]
        avg_lab_color = segment_pixels.mean(axis=0)
        
        # B. Normalized Area/Size - 1 feature
        area = np.sum(mask_flat) / total_pixels 
        
        # C. Centroid Location (Normalized Y/Vertical Position) - 1 feature
        # We use the centroid of the mask in the original image space
        mask = mask_flat.reshape((H, W)).astype(np.uint8)
        
        # Calculate moments for centroid
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cY = int(M["m01"] / M["m00"])
            # Normalize Y coordinate
            normalized_cY = cY / H
        else:
            # Fallback for tiny/empty segments
            normalized_cY = 0.5 

        # Combine features: [L, a, b, Area, Centroid_Y]
        segment_feature = np.concatenate([avg_lab_color, [area], [normalized_cY]])
        features.append(segment_feature)
        
    return np.array(features), labels.reshape((H, W)), kmeans.cluster_centers_

# --- II. SIMULATE TRAINING DATA (Replace with your actual dataset) ---

def create_simulated_dataset(num_samples=10, k_clusters=5):
    """
    Simulates extracting features from multiple images belonging to different classes.
    
    In a real scenario, you would loop through all training images, call 
    extract_features_from_segments(), and label the resulting features.
    
    Classification is done on segments, not whole images. The 'y' here is the 
    classification label for the WHOLE IMAGE (e.g., "Park", "Street").
    The K-NN classifier learns: "A picture with a large, dark, bottom segment 
    and a bright, top segment is likely a 'Street' scene."
    """
    print("--- Simulating Training Data ---")
    
    # For simulation, we create synthetic features for 2 classes: 'Park' (0) and 'Street' (1)
    
    # Example features for a 'Park' scene (Green/Sky/Dirt):
    # L, a, b, Area, Y_cent
    park_features = np.random.rand(num_samples * k_clusters // 2, 5)
    # Bias Park features: e.g., higher 'a' (green) in L*a*b*, larger Area for grass
    park_features[:, 1] += 100 
    park_features[:, 3] += 0.5 
    park_labels = np.full(num_samples * k_clusters // 2, 0) # Class 0: Park

    # Example features for a 'Street' scene (Gray/Sky/Road):
    street_features = np.random.rand(num_samples * k_clusters // 2, 5)
    # Bias Street features: e.g., low 'a' (gray), larger Area for road segment, low Y_cent for road segment
    street_features[:, 1] *= 0.1 
    street_features[:, 3] += 0.3
    street_features[:, 4] -= 0.5 
    street_labels = np.full(num_samples * k_clusters // 2, 1) # Class 1: Street

    X = np.concatenate([park_features, street_features])
    y = np.concatenate([park_labels, street_labels])
    
    # IMPORTANT: The 'y' label for K-NN must be the IMAGE CLASS, 
    # and all segments from one image share that same class label.
    
    # Since we are predicting the class of the whole image, we need to map the 
    # features back to their original image ID for correct cross-validation.
    # For this simple K-NN on features, we skip this complex step and rely 
    # on the K-NN voting mechanism to classify the new image.
    
    return X, y, {0: "Park Scene", 1: "Street Scene"}


# --- III. CLASSIFICATION WORKFLOW ---

def classify_segmented_image(test_image_path, k_clusters=5, n_neighbors=3):
    """
    Trains K-NN on simulated data and classifies a new test image based on its segments.
    """
    
    # 1. Prepare Training Data
    X_train, y_train, class_map = create_simulated_dataset(k_clusters=k_clusters)

    # 2. Initialize and Train K-NN Classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    print(f"K-NN Model Trained with {len(X_train)} segment features.")

    # 3. Extract Features from the Test Image
    img_bgr = cv2.imread(test_image_path)
    if img_bgr is None:
        print(f"Error: Could not load image at {test_image_path}")
        return
        
    X_test_segments, labels_test, centers_lab = extract_features_from_segments(img_bgr, k_clusters)
    
    # 4. Classify Segments
    # Each segment feature vector is classified independently
    predictions = knn.predict(X_test_segments)
    
    # 5. Determine Final Image Class by Majority Vote
    # Count the votes for each class among the segments
    unique, counts = np.unique(predictions, return_counts=True)
    vote_counts = dict(zip(unique, counts))
    
    final_class_id = unique[np.argmax(counts)]
    final_class_name = class_map[final_class_id]
    
    print("\n--- CLASSIFICATION RESULTS ---")
    print(f"Segment Predictions: {vote_counts}")
    print(f"Final Image Classification (Majority Vote): **{final_class_name}**")
    
    # 6. Visualization (Segmented image mapped to predicted class colors)
    # We use the final image class's 'dominant' color for a simple visualization
    
    # Get the RGB color from the L*a*b* center closest to the predicted class's 
    # general color (This step is complex, so we'll just use a generic color for the image class)
    if final_class_id == 0:
        vis_color = (0, 150, 0) # Green for Park
    else:
        vis_color = (150, 0, 0) # Red for Street
        
    vis_img = np.full(img_bgr.shape, vis_color, dtype=np.uint8)
    vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.title("Original Test Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(vis_img_rgb)
    plt.title(f"Classified as: {final_class_name}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# --- Usage Example ---
# IMPORTANT: Replace 'test_image.jpg' with a real image path 
# and ensure it's a scene (like a park or a street) to match the simulated data.

classify_segmented_image('extracted_frame.jpg', k_clusters=5, n_neighbors=5)