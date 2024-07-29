import os
import sys
import cv2
import mediapipe as mp
import numpy as np
import pygame
from threading import Thread
import time
import csv
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Ensure the module for YOLOv10 is accessible
yolov10_path = os.path.join('..', 'yolov10')
sys.path.append(yolov10_path)

# Install dependencies
os.system('pip install huggingface_hub -i https://mirrors.cloud.tencent.com/pypi/simple')
os.system(
    f'pip install -r {os.path.join(yolov10_path, "requirements.txt")} -i https://mirrors.cloud.tencent.com/pypi/simple')
os.system(f'pip install -e {yolov10_path} -i https://mirrors.cloud.tencent.com/pypi/simple')

from ultralytics import YOLOv10

# Load the YOLOv10 model
model_file_path = os.path.join('..', 'model', 'pp_table_net.pt')
model = YOLOv10(model_file_path)

# Initialize Pygame
pygame.init()

# Set the width and height of the screen [width, height]
screen_width, screen_height = 1280, 720
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Pose Estimation and Analysis")

# Define colors
WHITE = (255, 255, 255)

# Setup the clock for a decent framerate
clock = pygame.time.Clock()
FPS = 30

class PoseEstimation:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.templates = {"Arm": [], "Footwork": []}
        self.recording = False
        self.keypoints_data = []
        self.video_playing = False
        self.video_length = 0
        self.current_frame = 0
        self.pingpong_class = 15
        self.cap = None
        self.TEMPLATES_FILE = 'templates.csv'
        self.dragging = False
        self.video_path = os.path.join('..', 'mp4', '01.mov')
        self.load_templates_from_csv()
        self.reset_variables()
        self.grid_rects = None
        self.show_overlay = False
        self.camera_params = None
        self.red_cross_coords = None  # 初始化 red_cross_coords
        self.load_chessboard_pattern_config()  # 初始化时加载棋盘配置
        self.covered_area = set()  # 初始化covered_area
        self.highlight_counts = {}  # 初始化highlight_counts
        self.large_square_width = 100.0  # 大格子的宽度（厘米）
        self.large_square_height = 75.0  # 大格子的高度（厘米）

    def reset_variables(self):
        self.previous_midpoint = None
        self.previous_foot_points = None
        self.previous_hand_points = None
        self.previous_time = None
        self.start_time = time.time()
        self.covered_area = set()  # 重置覆盖区域
        self.highlight_counts = {}  # 重置高亮次数统计
        self.speeds = {
            'forward': [],
            'sideways': [],
            'depth': [],
            'overall': []
        }  # 重置速度统计
        self.template_match_counts = {"Arm": {}, "Footwork": {}}  # 重置模板匹配计数
        self.last_matched_templates = {"Arm": set(), "Footwork": set()}  # 重置最后匹配的模板

    def load_templates_from_csv(self):
        self.templates = {"Arm": [], "Footwork": []}
        if os.path.exists(self.TEMPLATES_FILE):
            try:
                with open(self.TEMPLATES_FILE, mode='r') as file:
                    reader = csv.reader(file)
                    next(reader)
                    for row in reader:
                        name = row[0]
                        category = row[1]
                        data = eval(row[2])
                        self.templates[category].append({'name': name, 'data': data})
            except (IOError, csv.Error) as e:
                print(f"Failed to load templates from CSV: {e}")

    def process_video(self, frame, pose):
        match_results = {"Arm": {}, "Footwork": {}}
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Update global variables for image dimensions
        self.image_width = image.shape[1]
        self.image_height = image.shape[0]

        # Add point (0,0) to the video frame
        cv2.circle(image, (0, 0), 20, (0, 255, 0), -1)

        # Add the maximum point (image.shape[1]-1, image.shape[0]-1) to the video frame
        max_point = (image.shape[1] - 1, image.shape[0] - 1)
        cv2.circle(image, max_point, 20, (255, 0, 0), -1)

        # Process pose landmarks
        keypoints = []
        foot_points = []
        hand_points = []
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            keypoints = [(lm.x, lm.y, lm.z) for lm in landmarks]

            # Extract foot and hand keypoints
            foot_points = [(landmarks[idx].x, landmarks[idx].y) for idx in [29, 31, 30, 32]]
            hand_points = [(landmarks[idx].x, landmarks[idx].y) for idx in [17, 19, 18, 20]]

        # Draw skeleton
        for idx, point in enumerate(keypoints):
            x, y, z = point
            cv2.circle(image, (int(x * self.image_width), int(y * self.image_height)), 5, (0, 255, 0), -1)

        return image

    def analyze_video(self):
        self.cap = cv2.VideoCapture(self.video_path)
        self.video_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.video_playing = True
        self.start_time = time.time()

        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while self.cap.isOpened() and self.video_playing:
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.current_frame += 1
                processed_image = self.process_video(frame, pose)
                self.display_frame(processed_image)
                clock.tick(FPS)

    def display_frame(self, frame):
        # Convert frame to pygame surface
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        frame = pygame.surfarray.make_surface(frame)
        screen.blit(frame, (0, 0))
        pygame.display.update()

    def save_templates_to_csv(self):
        with open(self.TEMPLATES_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['name', 'category', 'data'])
            for category, templates in self.templates.items():
                for template in templates:
                    writer.writerow([template['name'], category, template['data']])

    def calculate_chessboard_data(self, frame,
                                  small_chessboard_size=(8, 8),
                                  small_square_size=10.0,
                                  large_square_width=100.0,
                                  large_square_height=75.0,
                                  vertical_offset=-15.0,
                                  num_large_squares_x=3,
                                  num_large_squares_y=6):
        def clamp_point(point, width, height):
            x, y = point
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))
            return x, y

        if frame is None:
            raise ValueError("输入帧为空，请检查图像路径是否正确")

        height, width, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, small_chessboard_size, None)

        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        else:
            raise ValueError("无法检测到棋盘格角点")

        objp = np.zeros((small_chessboard_size[0] * small_chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:small_chessboard_size[0], 0:small_chessboard_size[1]].T.reshape(-1, 2)
        objp *= small_square_size

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp], [corners], gray.shape[::-1], None, None)
        left_bottom_corner = corners[-1][0]
        chessboard_physical_points = []

        for i in range(num_large_squares_y + 1):
            for j in range(num_large_squares_x + 1):
                chessboard_physical_points.append([j * large_square_width, -i * large_square_height - vertical_offset, 0])

        chessboard_physical_points = np.array(chessboard_physical_points, dtype=np.float32)

        def project_points(physical_points, rvec, tvec, mtx, dist):
            image_points, _ = cv2.projectPoints(physical_points, rvec, tvec, mtx, dist)
            return image_points.reshape(-1, 2)

        chessboard_image_points_px = project_points(chessboard_physical_points, rvecs[0], tvecs[0], mtx, dist)
        chessboard_vertices = []

        for i in range(num_large_squares_y):
            for j in range(num_large_squares_x):
                top_left = chessboard_image_points_px[i * (num_large_squares_x + 1) + j]
                top_right = chessboard_image_points_px[i * (num_large_squares_x + 1) + j + 1]
                bottom_right = chessboard_image_points_px[(i + 1) * (num_large_squares_x + 1) + j + 1]
                bottom_left = chessboard_image_points_px[(i + 1) * (num_large_squares_x + 1) + j]
                chessboard_vertices.append([top_left, top_right, bottom_right, bottom_left])

        p1 = chessboard_physical_points[0]
        p2 = chessboard_physical_points[1]
        p3 = chessboard_physical_points[num_large_squares_x + 1]
        v1 = p2 - p1
        v2 = p3 - p1
        normal_vector = np.cross(v1, v2)
        normal_vector = normal_vector / np.linalg.norm(normal_vector)
        middle_right_square_idx = (num_large_squares_y // 2) * num_large_squares_x + (num_large_squares_x - 1)
        right_top_vertex_idx = (num_large_squares_x + 1) * (middle_right_square_idx // num_large_squares_x) + (
                middle_right_square_idx % num_large_squares_x) + 1
        right_top_vertex_phys = chessboard_physical_points[right_top_vertex_idx]

        error_ratio = 0.8
        normal_vector = normal_vector * error_ratio
        vertical_end_point_phys = right_top_vertex_phys + normal_vector * 76

        right_top_vertex_img, _ = cv2.projectPoints(right_top_vertex_phys.reshape(1, 1, 3), rvecs[0], tvecs[0], mtx, dist)
        vertical_end_point_img, _ = cv2.projectPoints(vertical_end_point_phys.reshape(1, 1, 3), rvecs[0], tvecs[0], mtx,
                                                      dist)
        right_top_vertex_img = tuple(map(int, right_top_vertex_img.reshape(2)))
        vertical_end_point_img = tuple(map(int, vertical_end_point_img.reshape(2)))

        horizontal_start_point_phys = vertical_end_point_phys - np.array([0, 76, 0], dtype=np.float32)
        horizontal_end_point_phys = vertical_end_point_phys + np.array([0, 76, 0], dtype=np.float32)

        horizontal_start_point_img, _ = cv2.projectPoints(horizontal_start_point_phys.reshape(1, 1, 3), rvecs[0], tvecs[0],
                                                          mtx, dist)
        horizontal_end_point_img, _ = cv2.projectPoints(horizontal_end_point_phys.reshape(1, 1, 3), rvecs[0], tvecs[0], mtx,
                                                        dist)
        horizontal_start_point_img = tuple(map(int, horizontal_start_point_img.reshape(2)))
        horizontal_end_point_img = tuple(map(int, horizontal_end_point_img.reshape(2)))

        vertical_line_1_phys = horizontal_start_point_phys - np.array([0, 0, -76 * error_ratio], dtype=np.float32)
        vertical_line_2_phys = horizontal_end_point_phys - np.array([0, 0, -76 * error_ratio], dtype=np.float32)

        vertical_line_1_end_img, _ = cv2.projectPoints(vertical_line_1_phys.reshape(1, 1, 3), rvecs[0], tvecs[0], mtx, dist)
        vertical_line_2_end_img, _ = cv2.projectPoints(vertical_line_2_phys.reshape(1, 1, 3), rvecs[0], tvecs[0], mtx, dist)
        vertical_line_1_end_img = tuple(map(int, vertical_line_1_end_img.reshape(2)))
        vertical_line_2_end_img = tuple(map(int, vertical_line_2_end_img.reshape(2)))

        height, width, _ = frame.shape

        normalized_chessboard_vertices = []
        for vertices in chessboard_vertices:
            normalized_vertices = [(pt[0] / width, pt[1] / height) for pt in vertices]
            normalized_chessboard_vertices.append(normalized_vertices)

        chessboard_data = {
            'chessboard_vertices': normalized_chessboard_vertices,
            'right_top_vertex_img': (right_top_vertex_img[0] / width, right_top_vertex_img[1] / height),
            'vertical_end_point_img': (vertical_end_point_img[0] / width, vertical_end_point_img[1] / height),
            'horizontal_start_point_img': (horizontal_start_point_img[0] / width, horizontal_start_point_img[1] / height),
            'horizontal_end_point_img': (horizontal_end_point_img[0] / width, horizontal_end_point_img[1] / height),
            'vertical_line_1_end_img': (vertical_line_1_end_img[0] / width, vertical_line_1_end_img[1] / height),
            'vertical_line_2_end_img': (vertical_line_2_end_img[0] / width, vertical_line_2_end_img[1] / height),
            'mtx': mtx,
            'dist': dist,
            'rvecs': rvecs[0],
            'tvecs': tvecs[0]
        }

        return chessboard_data

    def draw_chessboard_on_frame(self, frame, chessboard_data, show_overlay=False, covered_area=None, highlight_ratios=None):
        if not show_overlay:
            return frame

        height, width, _ = frame.shape
        output_image = frame.copy()

        norm = mcolors.Normalize(vmin=0, vmax=100)
        cmap = plt.get_cmap('hot')

        for idx, vertices in enumerate(chessboard_data['chessboard_vertices']):
            pts = np.array([(int(pt[0] * width), int(pt[1] * height)) for pt in vertices], dtype=np.int32)

            vertex_tuples = tuple(map(tuple, vertices))
            if highlight_ratios and vertex_tuples in highlight_ratios:
                color = cmap(norm(highlight_ratios[vertex_tuples]))[:3]
                color = tuple(int(c * 255) for c in color)
                cv2.fillPoly(output_image, [pts], color=color)

            cv2.polylines(output_image, [pts], isClosed=True, color=(0, 255, 0), thickness=1)

            center_x = int(np.mean([pt[0] for pt in pts]))
            center_y = int(np.mean([pt[1] for pt in pts]))

            text = str(idx + 1)
            special_texts = {9: "R00", 6: "R01", 3: "R02", 8: "R10", 5: "R11", 2: "R12", 7: "R20", 4: "R21", 1: "R22",
                             12: "L00", 15: "L01", 18: "L02", 11: "L10", 14: "L11", 17: "L12", 10: "L20", 13: "L21",
                             16: "L22"}
            if idx + 1 in special_texts:
                text = special_texts[idx + 1]

            cv2.putText(output_image, text, (center_x, center_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if highlight_ratios and vertex_tuples in highlight_ratios and highlight_ratios[vertex_tuples] > 0:
                cv2.polylines(output_image, [pts], isClosed=True, color=(0, 0, 255), thickness=1)

        if 'right_top_vertex_img' in chessboard_data and 'vertical_end_point_img' in chessboard_data:
            right_top_vertex_img = (int(chessboard_data['right_top_vertex_img'][0] * width),
                                    int(chessboard_data['right_top_vertex_img'][1] * height))
            vertical_end_point_img = (int(chessboard_data['vertical_end_point_img'][0] * width),
                                      int(chessboard_data['vertical_end_point_img'][1] * height))
            cv2.line(output_image, right_top_vertex_img, vertical_end_point_img, (255, 255, 255), 1)

        if 'horizontal_start_point_img' in chessboard_data and 'horizontal_end_point_img' in chessboard_data:
            horizontal_start_point_img = (int(chessboard_data['horizontal_start_point_img'][0] * width),
                                          int(chessboard_data['horizontal_start_point_img'][1] * height))
            horizontal_end_point_img = (int(chessboard_data['horizontal_end_point_img'][0] * width),
                                        int(chessboard_data['horizontal_end_point_img'][1] * height))
            cv2.line(output_image, horizontal_start_point_img, horizontal_end_point_img, (255, 255, 255), 1)

        if 'vertical_line_1_end_img' in chessboard_data and 'vertical_line_2_end_img' in chessboard_data:
            vertical_line_1_end_img = (int(chessboard_data['vertical_line_1_end_img'][0] * width),
                                       int(chessboard_data['vertical_line_1_end_img'][1] * height))
            vertical_line_2_end_img = (int(chessboard_data['vertical_line_2_end_img'][0] * width),
                                       int(chessboard_data['vertical_line_2_end_img'][1] * height))
            cv2.line(output_image, horizontal_start_point_img, vertical_line_1_end_img, (255, 255, 255), 1)
            cv2.line(output_image, horizontal_end_point_img, vertical_line_2_end_img, (255, 255, 255), 1)

        return output_image

    def load_chessboard_pattern_config(self):
        if os.path.exists("chessboard_pattern_config.json"):
            try:
                with open("chessboard_pattern_config.json", "r") as f:
                    config = json.load(f)
                    self.grid_rects = config["grid_rects"]
                    self.red_cross_coords = config["red_cross_coords"]
                    camera_params = config["camera_params"]
                    self.camera_params = (
                        np.array(camera_params["mtx"]),
                        np.array(camera_params["dist"]),
                        [np.array(rvec) for rvec in camera_params["rvecs"]],
                        [np.array(tvec) for tvec in camera_params["tvecs"]]
                    )
            except Exception as e:
                print(f"Failed to load chessboard pattern config: {e}")
                self.camera_params = None
                self.grid_rects = None
                self.red_cross_coords = None


class PoseApp:
    def __init__(self, pose_estimation):
        self.pose_estimation = pose_estimation
        self.calculate_chessboard = False
        self.pose_estimation.app = self
        self.current_layout = 1

        # Start with analyzing video
        self.mode = "match_template"
        self.update_mode_label()
        self.pose_estimation.analyze_video()

    def update_mode_label(self):
        self.pose_estimation.reset_variables()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.pose_estimation.close_camera()
                pygame.quit()
                sys.exit()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.pose_estimation.close_camera()
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_a:
                    self.pose_estimation.close_camera()
                    self.mode = "analyze_video"
                    self.update_mode_label()
                    self.pose_estimation.analyze_video()
                elif event.key == pygame.K_b:
                    self.pose_estimation.recording = True
                    print("Recording started")
                elif event.key == pygame.K_e:
                    self.pose_estimation.recording = False
                    print("Recording stopped")
                    if self.pose_estimation.keypoints_data:
                        input_dialog = TemplateInputDialog()
                        template_name, category = input_dialog.get_input()
                        if template_name and category:
                            self.pose_estimation.templates[category].append(
                                {"name": template_name, "data": self.pose_estimation.keypoints_data})
                            self.pose_estimation.keypoints_data = []
                            self.pose_estimation.save_templates_to_csv()
                elif event.key == pygame.K_F5:
                    self.pose_estimation.close_camera()
                    self.mode = "real_time"
                    self.update_mode_label()
                    self.pose_estimation.start_real_time_analysis()
                elif event.key == pygame.K_F6:
                    if any(self.pose_estimation.templates.values()):
                        self.pose_estimation.close_camera()
                        self.mode = "match_template"
                        self.update_mode_label()
                        self.pose_estimation.analyze_video()
                elif event.key == pygame.K_F1:
                    chessboard_params = {
                        "small_chessboard_size": (8, 8),
                        "small_square_size": 10.0,
                        "large_square_width": 100.0,
                        "large_square_height": 75.0,
                        "vertical_offset": -15.0,
                        "num_large_squares_x": 3,
                        "num_large_squares_y": 6
                    }
                    self.pose_estimation.save_chessboard_pattern(chessboard_params, self.pose_estimation.grid_rects,
                                                                 self.pose_estimation.red_cross_coords,
                                                                 self.pose_estimation.camera_params)
                elif event.key == pygame.K_F2:
                    self.calculate_chessboard = True

                elif event.key == pygame.K_F3:
                    self.pose_estimation.show_overlay = not self.pose_estimation.show_overlay

                elif event.key == pygame.K_F4:
                    self.current_layout = 2 if self.current_layout == 1 else 1


class TemplateInputDialog:
    def __init__(self):
        self.template_name = None
        self.category = None

    def get_input(self):
        self.template_name = input("Enter template name: ")
        self.category = input("Enter template category (Arm/Footwork): ")
        return self.template_name, self.category


def main():
    pose_estimation = PoseEstimation()
    app = PoseApp(pose_estimation)

    running = True
    while running:
        app.handle_events()
        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
