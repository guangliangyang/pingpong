import cv2
import numpy as np
import time
from ultralytics import YOLO
from determine_region import determine_region
import torch

# Initialize the YOLO model
model = YOLO('best_bak.pt')
confidence_threshold = 0.25

# Provide the path to your video file
video_path = 'Yan2023.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)

camera_matrix = np.load('camera_matrix.npy')
dist_coeffs = np.load('dist_coeffs.npy')
rvecs = np.load('rvecs.npy')
tvecs = np.load('tvecs.npy')

# Scaling factor for converting pixels to meters
square_size_pixels = 160  # Known size in pixels
square_size = 100  # mm
scaling_factor = square_size / square_size_pixels

# Create a background subtractor object with threshold
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25, detectShadows=False)

# Initialize variables for previous frame
prev_boxes = None
prev2_boxes = None
prev_time = time.time()

# Initialize region data
region_data_L = [['L1', 0], ['L2', 0], ['L3', 0], ['L4', 0], ['L5', 0],
               ['L6', 0], ['L7', 0], ['L8', 0], ['L9', 0]]
region_data_R = [['R1', 0], ['R2', 0], ['R3', 0], ['R4', 0], ['R5', 0], ['R6', 0],
               ['R7', 0], ['R8', 0], ['R9', 0]]
proportion_region_data_L = [['L1', 0], ['L2', 0], ['L3', 0], ['L4', 0], ['L5', 0],
               ['L6', 0], ['L7', 0], ['L8', 0], ['L9', 0]]
proportion_region_data_R = [['R1', 0], ['R2', 0], ['R3', 0], ['R4', 0], ['R5', 0], ['R6', 0],
               ['R7', 0], ['R8', 0], ['R9', 0]]

# Initialize a frame counter
frame_counter = 0

# Initialize speed data
speed = 0

# Function to rotate a frame by 90 degrees
def rotate_frame(frame):
    return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

# Function to flip the frame horizontally
def flip_frame(frame):
    return cv2.flip(frame, 1)

# Run the main loop
while True:
    # Capture a frame from the video
    success, frame = cap.read()

    if not success:
        break

    # Apply background subtraction
    foreground_mask = bg_subtractor.apply(frame)

    # Perform morphological operations (noise removal)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)

    # Convert foreground mask to RGB for YOLOv8 input
    foreground_rgb = cv2.cvtColor(foreground_mask, cv2.COLOR_GRAY2RGB)

    # Run YOLOv8 inference on the frame
    results = model(foreground_rgb, conf=confidence_threshold, device='1')

    # Initialize variables for current frame
    curr_boxes = None
    curr_time = time.time()
    all_boxes = []

    for result in results:
        if len(result) == 0:  # No detections in this result
            continue
        boxes_tensor = result.boxes.data
        boxes = boxes_tensor.cpu().numpy()
        all_boxes.append(boxes)

    if len(all_boxes) > 0:
        all_boxes = np.vstack(all_boxes)
        class_3_boxes = all_boxes[all_boxes[:, 5] == 3]
        if len(class_3_boxes) > 0:
            sorted_class_3_boxes = class_3_boxes[class_3_boxes[:, 1].argsort()]
            curr_boxes = np.array([sorted_class_3_boxes[0]])

    try:
        if prev_boxes is not None and curr_boxes is not None and prev2_boxes is not None:
            prev_y = prev_boxes[0][1]
            prev2_y = prev2_boxes[0][1]

            for box in curr_boxes:
                for prev_box in prev_boxes:
                    if prev2_y < prev_box[1] and box[1] < prev_box[1]:
                        region = determine_region(prev_box)
                        if region:
                            for idx, data in enumerate(region_data_L):
                                if data[0] == region:
                                    region_data_L[idx][1] += 1
                                    total_L = sum(item[1] for item in region_data_L)
                                    for i in range(len(proportion_region_data_L)):
                                        proportion_L = (region_data_L[i][1] / total_L) * 100
                                        proportion_region_data_L[i][1] = "{:.2f}%".format(proportion_L)

                            for idx, data in enumerate(region_data_R):
                                if data[0] == region:
                                    region_data_R[idx][1] += 1
                                    total_R = sum(item[1] for item in region_data_R)
                                    for i in range(len(proportion_region_data_R)):
                                        proportion_R = (region_data_R[i][1] / total_R) * 100
                                        proportion_region_data_R[i][1] = "{:.2f}%".format(proportion_R)

    except Exception as e:
        print("Error occurred:", str(e))

    try:
        if prev_boxes is not None and curr_boxes is not None:
            curr_boxes_sorted = curr_boxes[curr_boxes[:, 4].argsort()[::-1]]
            curr_box = curr_boxes_sorted[0]
            prev_center = (prev_boxes[:, :2] + prev_boxes[:, 2:4]) / 2
            curr_center = (curr_boxes[:, :2] + curr_boxes[:, 2:4]) / 2

            prev_points = cv2.undistortPoints(np.expand_dims(prev_center, axis=1), camera_matrix, dist_coeffs, None, camera_matrix)
            curr_points = cv2.undistortPoints(np.expand_dims(curr_center, axis=1), camera_matrix, dist_coeffs, None, camera_matrix)
            prev_points_3d = cv2.convertPointsToHomogeneous(prev_points)
            curr_points_3d = cv2.convertPointsToHomogeneous(curr_points)
            R, _ = cv2.Rodrigues(rvecs)
            T = tvecs.reshape((3, 1))
            prev_points_3d_cam = np.matmul(R, prev_points_3d.transpose(0, 2, 1)) + T
            curr_points_3d_cam = np.matmul(R, curr_points_3d.transpose(0, 2, 1)) + T
            displacement_3d = np.squeeze(curr_points_3d_cam - prev_points_3d_cam)
            distances = np.linalg.norm(displacement_3d)
            distance_real = distances * scaling_factor * 0.036
            time_diff = curr_time - prev_time
            if time_diff == 0:
                speed_str = "speed: 0km/h"
            else:
                speed = np.abs(distance_real) / time_diff
                speed_str = 'speed: {:.2f}km/h'.format(speed)
        else:
            speed_str = "speed: 0km/h"

    except Exception as e:
        print("Error occurred:", str(e))

    # Update variables for next frame
    if prev_boxes is not None:
        prev2_boxes = prev_boxes
        prev_boxes = curr_boxes
        prev_time = curr_time
    else:
        prev_boxes = curr_boxes
        prev_time = curr_time

# Release the video capture object
cap.release()

# Print the statistics
print("Ball Speed:", speed_str)
print("Proportion Region Data L:")
for data in proportion_region_data_L:
    print(f"{data[0]}: {data[1]}")
print("Proportion Region Data R:")
for data in proportion_region_data_R:
    print(f"{data[0]}: {data[1]}")
