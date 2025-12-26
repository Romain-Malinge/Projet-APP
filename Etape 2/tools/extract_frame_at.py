import cv2
import math

def save_frame_at_nanoseconds(video_path, output_path, target_ns):
    """
    Extracts and saves a frame from a video at a specific timestamp in nanoseconds.
    
    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path to save the extracted image (e.g., 'frame.jpg').
        target_ns (int): Target timestamp in nanoseconds.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps == 0:
        print("Error: FPS is 0, cannot calculate frame positions.")
        return

    # 1. Calculate the exact target frame number
    # Nanoseconds to seconds
    target_seconds = target_ns / 1_000_000_000.0
    target_frame_number = int(round(target_seconds * fps))

    print(f"Target Time: {target_seconds}s")
    print(f"Target Frame Number: {target_frame_number}")

    if target_frame_number >= total_frames:
        print(f"Error: Target frame {target_frame_number} exceeds total frames {total_frames}.")
        return

    # 2. Seek strategy for accuracy
    # OpenCV's set(CAP_PROP_POS_FRAMES) is fast but sometimes lands on 
    # the nearest KeyFrame (I-Frame) depending on the codec/backend.
    # To be safe, we seek to the target and then check where we actually landed.
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_number)
    
    # Check the position we actually landed on
    current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    
    # If the seek overshot (rare but possible) or undershot significantly, 
    # we might need to handle it. Usually, standard seek is 'good enough' for 
    # simple extraction, but for perfect accuracy, we can verify:
    if current_frame_pos != target_frame_number:
        print(f"Warning: Seek landed on frame {current_frame_pos} instead of {target_frame_number}.")
        # If strict accuracy is required, one method is to seek to an earlier frame
        # and iterate using read() until the target is reached.
        # Uncomment below to enable strict mode (slower but more accurate):
        
        # cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target_frame_number - 30))
        # while cap.get(cv2.CAP_PROP_POS_FRAMES) < target_frame_number:
        #     cap.read()

    # 3. Read the frame
    success, frame = cap.read()
    
    if success:
        cv2.imwrite(output_path, frame)
        print(f"Success: Frame saved to {output_path}")
    else:
        print("Error: Could not read frame at the target position.")

    cap.release()

# --- Usage Example ---
# 15 seconds + 500 milliseconds = 15.5 seconds
# 15.5 * 1,000,000,000 = 15,500,000,000 nanoseconds
video_file = "Cyclistes/first/first.mp4"
output_file = "extracted_frame.jpg"
timestamp_ns =  ((1763649017075260000 + 1763649017345510000) / 2) - 1763649013947400000 

save_frame_at_nanoseconds(video_file, output_file, timestamp_ns)