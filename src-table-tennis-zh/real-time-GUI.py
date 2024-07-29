import pygame
import cv2


import time
import numpy as np
from ultralytics import YOLO
from determine_region import determine_region
import torch

# Initialize Pygame
pygame.init()

# Define the window size
window_width = 1530 #1440 #1280 #1530
window_height = 930#900 #930

# Create the Pygame window
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Scene Understanding of Table Tennis")

# Define the colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)

# Define the font
font = pygame.font.Font(None, 25)

# Define the layout dimensions
video_width = int(window_width * 0.8)
video_height = window_height

form_width = int(window_width * 0.2)
form_height = window_height

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Connect to the camera using the appropriate URL
#url = 'http://192.168.1.103:9372/video'  # Replace with your iPhone's IP address and the port number specified by the app
#cap = cv2.VideoCapture(url)

# Check if the camera stream is opened successfully
#if not cap.isOpened():
#    print("Failed to open the camera stream")
#    exit()

# Load the YOLOv8n model
model = YOLO('best_bak.pt')

confidence_threshold = 0.25


# Provide the path to your video file6174
video_path = 'Yan2023.mp4'


# Open the video file
cap = cv2.VideoCapture(video_path)


prev_frame = None  # Initialize prev_frame variable before the loop

# Initialize variables for previous frame
prev_boxes = None
prev2_boxes = None
prev_time = time.time()

# Define the class names
names = {0: 'net', 1: '1-1', 2: 'paddle', 3: 'Table tennis ball', 4: '1-2', 5: '1-4', 6: '1-5', 7: '2-5', 8: '0-0', 9: '1-0'}

camera_matrix = np.load('camera_matrix.npy')
dist_coeffs = np.load('dist_coeffs.npy')
rvecs = np.load('rvecs.npy')
tvecs = np.load('tvecs.npy')

# scaling factor for converting pixels to meters
square_size_pixels = 160  # Known size in pixels #60
square_size = 100  # mm

scaling_factor = square_size / square_size_pixels

# Create an empty list to store the lower bounding boxes
lower_boxes = []
finding_lower_boxes = False  # Flag to indicate if we are currently finding lower bounding boxes

all_lower_boxes = []

# Initialize a frame counter
frame_counter = 0

# Initialize speed data
speed = 0

# Initialize region data

region_data_L = [['L1', 0], ['L2', 0], ['L3', 0], ['L4', 0], ['L5', 0],
               ['L6', 0], ['L7', 0], ['L8', 0], ['L9', 0]]
region_data_R = [['R1', 0], ['R2', 0], ['R3', 0], ['R4', 0], ['R5', 0], ['R6', 0],
               ['R7', 0], ['R8', 0], ['R9', 0]]
proportion_region_data_L = [['L1', 0], ['L2', 0], ['L3', 0], ['L4', 0], ['L5', 0],
               ['L6', 0], ['L7', 0], ['L8', 0], ['L9', 0]]
proportion_region_data_R = [['R1', 0], ['R2', 0], ['R3', 0], ['R4', 0], ['R5', 0], ['R6', 0],
               ['R7', 0], ['R8', 0], ['R9', 0]]
# Function to rotate a frame by 90 degrees
def rotate_frame(frame):
    # Rotate the frame by 90 degrees clockwise
    return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

# Function to flip the frame horizontally
def flip_frame(frame):
    # Flip the frame horizontally (along the y-axis)
    return cv2.flip(frame, 1)

# Create a background subtractor object with threshold
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25, detectShadows=False)

# Run the main loop
running = True
while running:
    # Handle Pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the window
    window.fill(WHITE)

    # Capture a frame from the webcam
    success, frame = cap.read()

    if not success:
        break

    # Get the original dimensions of the webcam video
    webcam_width = frame.shape[1]
    webcam_height = frame.shape[0]
    print("webcam_width:",  webcam_width)
    print("webcam_height:", webcam_height)

    # Calculate the scale factors for maintaining the aspect ratio
    scale_x = webcam_width / video_width
    scale_y = webcam_height / video_height
    scale_factor = min(scale_x, scale_y)
    print("video_width:", video_width)
    print("video_height:", video_height)
    print("scale_x:", scale_x)
    print("scale_y:", scale_y)
    print("scale_factor:", scale_factor)



    if success:

        # Apply background subtraction
        foreground_mask = bg_subtractor.apply(frame)

        # Perform morphological operations (noise removal)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)

        # Convert foreground mask to RGB for YOLOv8 input
        foreground_rgb = cv2.cvtColor(foreground_mask, cv2.COLOR_GRAY2RGB)

        # Run YOLOv8 inference on the frame
        results = model(foreground_rgb, conf=confidence_threshold, device='0')

        # Initialize variables for current frame
        curr_boxes = None
        curr_time = time.time()
        all_boxes = []
        curr_labels = []

        for result in results:
            if len(result) == 0:  # No detections in this result
                continue
            boxes_tensor = result.boxes.data
            # Convert tensor to NumPy array
            boxes = boxes_tensor.cpu().numpy()
            print("boxes", boxes)
            curr_labels = result.boxes.cls
            # Extract the tensor from the list
            #first_element = boxes[0]  # Assuming you have only one tensor in the list
            #all_boxes.append(first_element)
            all_boxes.append(boxes)


            if len(all_boxes) > 0:
                all_boxes = np.vstack(all_boxes)
                print("all_boxes", all_boxes)
                # Filter out boxes with class label 3
                class_3_boxes = all_boxes[all_boxes[:, 5] == 3]
                if len(class_3_boxes) > 0:
                    #curr_boxes = class_3_boxes
                    # Sort class_3_boxes based on y-coordinate (index 1)
                    sorted_class_3_boxes = class_3_boxes[class_3_boxes[:, 1].argsort()]

                    # Keep the box with the smallest y-coordinate (the first row after sorting)
                    curr_boxes = np.array([sorted_class_3_boxes[0]])
    try:
        if prev_boxes is not None and curr_boxes is not None and prev2_boxes is not None:
            # Calculate the y-coordinate of the previous frame's bounding boxes
            prev_y = prev_boxes[0][1]  # Assuming y-coordinate is the 2nd column (index 1)
            print("prev_y", prev_y)
            # Extract the y-coordinate from the subsequent frame's bounding box
            prev2_y = prev2_boxes[0][1]  # Assuming y-coordinate is the 2nd column (index 1)
            print("prev2_y", prev2_y)
            print("curr_boxes", curr_boxes)
            # Create a dictionary to store the counts for each region
            region_counts = {region: 0 for region in range(1, 19)}

            # Find the bounding boxes in the current frame that have a lower y-coordinate than prev_y_avg
            for box in curr_boxes:
                for prev_box in prev_boxes:
                    if prev2_y < prev_box[1] and box[1] < prev_box[1]:  # Assuming y-coordinate is the 2nd column (index 1)
                        all_lower_boxes.append(prev_box.tolist())
                        # Determine the region for the current bounding box
                        region = determine_region(prev_box)  # Implement your logic to determine the region based on the bounding box coordinates

                        if region:
                            # Find the index of the region in region_data_L and update the count
                            for idx, data in enumerate(region_data_L):
                                if data[0] == region:
                                    region_data_L[idx][1] += 1
                                    total_L = sum(item[1] for item in
                                                  region_data_L)  # Calculate the total sum of second values in region_data_L

                                    # Update each second data in proportion_region_data_L with calculated proportions
                                    for i in range(len(proportion_region_data_L)):
                                        proportion_L = (region_data_L[i][1] / total_L) * 100
                                        proportion_region_data_L[i][1] = "{:.2f}%".format(proportion_L)

                                    print("proportion_region_data_L:", proportion_region_data_L)
                                    break
                            if region:
                                # Find the index of the region in region_data_R and update the count
                                for idx, data in enumerate(region_data_R):
                                    if data[0] == region:
                                        region_data_R[idx][1] += 1
                                        total_R = sum(item[1] for item in
                                                      region_data_R)  # Calculate the total sum of second values in region_data_R

                                        # Update each second data in proportion_region_data_R with calculated proportions
                                        for i in range(len(proportion_region_data_R)):
                                            proportion_R = (region_data_R[i][1] / total_R) * 100
                                            proportion_region_data_R[i][1] = "{:.2f}%".format(proportion_R)

                                        print("proportion_region_data_R:", proportion_region_data_R)
                                        break
                                for idx, data in enumerate(region_data_R):
                                    if data[0] == region:
                                        region_data_R[idx][1] += 1
                                        total_R = sum(item[1] for item in
                                                      region_data_R)  # Calculate the total sum of second values in region_data_R

                                        # Update each second data in proportion_region_data_R with calculated proportions
                                        for i in range(len(proportion_region_data_R)):
                                            proportion_R = (region_data_R[i][1] / total_R) * 100
                                            proportion_region_data_R[i][1] = "{:.2f}%".format(proportion_R)

                                        print("proportion_region_data_R:", proportion_region_data_R)
                                        break
                            # Print the updated region_data
                            for data in region_data_L or region_data_R:
                                print(f"{data[0]}: {data[1]} bounding boxes")
                            # Save all lower bounding boxes to a single text file
                            if len(all_lower_boxes) > 0:
                                np.savetxt('all_lower_boxes.txt', all_lower_boxes, delimiter=',', fmt='%.2f')
                            if prev_frame is not None:
                                cv2.imwrite(f'landing/frame_{frame_counter}.jpg', prev_frame)
                                frame_counter += 1
                            # Update the previous frame for the next iteration
                            prev_frame = frame.copy()
    except Exception as e:
        print("Error occurred:", str(e))

    try:
        if prev_boxes is not None and curr_boxes is not None:
            # Sort the bounding boxes by confidence (descending order)
            curr_boxes_sorted = curr_boxes[curr_boxes[:, 4].argsort()[::-1]]

            # Get the most confident bounding box
            curr_box = curr_boxes_sorted[0]
            # Calculate displacement between previous and current frame
            prev_center = (prev_boxes[:, :2] + prev_boxes[:, 2:4]) / 2

            curr_center = (curr_boxes[:, :2] + curr_boxes[:, 2:4]) / 2


            # Calculate 3D displacement
            prev_points = cv2.undistortPoints(np.expand_dims(prev_center, axis=1), camera_matrix, dist_coeffs, None,
                                              camera_matrix)
            curr_points = cv2.undistortPoints(np.expand_dims(curr_center, axis=1), camera_matrix, dist_coeffs, None,
                                              camera_matrix)
            prev_points_3d = cv2.convertPointsToHomogeneous(prev_points)
            curr_points_3d = cv2.convertPointsToHomogeneous(curr_points)
            R, _ = cv2.Rodrigues(rvecs)
            T = tvecs.reshape((3, 1))
            prev_points_3d_cam = np.matmul(R, prev_points_3d.transpose(0, 2, 1)) + T
            curr_points_3d_cam = np.matmul(R, curr_points_3d.transpose(0, 2, 1)) + T
            displacement_3d = np.squeeze(curr_points_3d_cam - prev_points_3d_cam)
            # Calculate the Euclidean distance for each displacement vector （euclidean metric)(pixel)
            distances = np.linalg.norm(displacement_3d)
            distance_real = distances * scaling_factor * 0.036 # mm/s to km/h
            print(distance_real)
            # Calculate speed of the object
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

    # Draw the form on the right side
    pygame.draw.rect(window, GRAY, (video_width, 0, form_width, form_height))

    # Draw the header
    header_text = pygame.font.SysFont(None, 30).render("Table Tennis", True, BLACK)
    window.blit(header_text, (video_width + 10, 10))

    # Draw the rows and columns
    row_height = (form_height - 80) // 18
    column_width = (form_width - 40) // 2

    # Draw the "Ball Speed" row
    ball_speed_text = pygame.font.SysFont(None, 25).render("Ball Speed:", True, BLACK)
    window.blit(ball_speed_text, (video_width + 10, 35))

    ball_speed_value = pygame.font.SysFont(None, 25).render("{:.2f}km/h".format(speed), True, BLACK)
    window.blit(ball_speed_value, (video_width + 20 + column_width + 10, 35))

    for i in range(1, 18):
        row_y = 60
        pygame.draw.rect(window, WHITE, (video_width + 10, row_y, column_width * 2 + 20, window_height - 70))
        #pygame.draw.rect(window, WHITE, (video_width + 20 + column_width, row_y, column_width, row_height))

        # Display the form
        form_x = window_width * 0.7 + 20
        form_y = 100
        row_height = 40
        text_margin = 10

    # Draw the head of form
    Region_text = pygame.font.SysFont(None, 25).render("Region", True, BLACK)
    window.blit(Region_text, (video_width + 15, 65))

    proportion_value = pygame.font.SysFont(None, 25).render("Probability", True, BLACK)
    window.blit(proportion_value, (video_width + 15 + column_width, 65))

    # Define text mappings for L and R labels
    L_text_mapping = ['L22', 'L21', 'L20', 'L12', 'L11', 'L10', 'L02', 'L01', 'L00']
    R_text_mapping = ['R00', 'R01', 'R02', 'R10', 'R11', 'R12', 'R20', 'R21', 'R22']

    # Draw the L column
    for i in range(9):
        L_text = L_text_mapping[i]
        text_surface = font.render(L_text, True, (15, 82, 186))
        window.blit(text_surface, (video_width + 35, form_y + i * row_height * 1.1))

    # Draw the R column
    for i in range(9):
        R_text = R_text_mapping[i]
        text_surface = font.render(R_text, True, (243, 112, 33))
        window.blit(text_surface, (video_width + 35, form_y + (9 + i) * row_height * 1.1))

    # Draw the second column proportion values and bars
    for i in range(len(proportion_region_data_L)):
        text_L = str(proportion_region_data_L[i][1])
        text_surface_L = font.render(text_L, True, BLACK)
        window.blit(text_surface_L, (video_width + 20 + column_width, form_y + i * row_height * 1.1))
        try:
            # Check if the value is a string and strip the '%' if so
            if isinstance(proportion_region_data_L[i][1], str):
                proportion_L = float(proportion_region_data_L[i][1].strip('%')) / 100
            else:
                proportion_L = float(proportion_region_data_L[i][1]) / 100
        except ValueError:
            proportion_L = 0
        bar_width_L = int(proportion_L * (column_width - text_margin * 2))
        bar_height = row_height * 1.1 - text_margin * 2
        bar_x_L = video_width + 20 + column_width
        bar_y_L = form_y + i * row_height * 1.1 + text_margin + 7
        pygame.draw.rect(window, (95, 142, 193), (bar_x_L, bar_y_L, bar_width_L, bar_height))

    for j in range(len(proportion_region_data_R)):
        text_R = str(proportion_region_data_R[j][1])
        text_surface_R = font.render(text_R, True, BLACK)
        window.blit(text_surface_R,
                    (video_width + 20 + column_width, form_y + j * row_height * 1.1 + (i + 1) * row_height * 1.1))
        try:
            # Check if the value is a string and strip the '%' if so
            if isinstance(proportion_region_data_R[j][1], str):
                proportion_R = float(proportion_region_data_R[j][1].strip('%')) / 100
            else:
                proportion_R = float(proportion_region_data_R[j][1]) / 100
        except ValueError:
            proportion_R = 0
        bar_width_R = int(proportion_R * (column_width - text_margin * 2))
        bar_x_R = video_width + 20 + column_width
        bar_y_R = form_y + j * row_height * 1.1 + text_margin + 7 + (i + 1) * row_height * 1.1
        bar_height = row_height * 1.1 - text_margin * 2
        pygame.draw.rect(window, (255, 165, 0), (bar_x_R, bar_y_R, bar_width_R, bar_height))

        # Set the bar color based on the row index
        #bar_color = (255, 165, 0) if 10 <= i <= 17 else (
        #95, 142, 193)  # (95, 142, 193) for rows 10 to 17, (255, 165, 0) for others (orange)


    # Rotate the frame by 90 degrees clockwise
    #frame = np.rot90(frame)
    # Get frame width and height
    height, width = frame.shape[:2]
    print("width", width)
    print("height", height)
    # Resize the frame with scale invariance
    resized_frame = cv2.resize(frame, (int(webcam_width * 1.9), int(webcam_height * 1.9))) #1.9 #0.84 for AUT 0.56 for video
    # Convert the frame to RGB format for Pygame
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Rotate the frame by 90 degrees
    rotated_frame = rotate_frame(rgb_frame)

    # Flip the frame horizontally
    flipped_frame = flip_frame(rotated_frame)

    # Flip the frame again to show normal view
    #flipped_frame = cv2.flip(flipped_frame, 0)

    # Create a Pygame surface from the frame
    frame_surface = pygame.surfarray.make_surface(flipped_frame)

    # Blit the frame onto the window
    window.blit(frame_surface, (4, 8))
    # Update the display
    pygame.display.update()

# Release the webcam
cap.release()

# Quit Pygame
pygame.quit()
