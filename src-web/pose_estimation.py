import os
import sys
import threading
import logging
import cv2
import mediapipe as mp
import numpy as np
import csv
import time
import json
import matplotlib.colors as mcolors

import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')
warnings.filterwarnings("ignore", category=UserWarning, module='inference_feedback_manager')

SETUP_YOLOV10 = False
try:
    from ultralytics import YOLOv10 as YOLO
except ImportError:
    print("YOLOv10 not found, setting up YOLOv10...")
    SETUP_YOLOV10 = True

if SETUP_YOLOV10:
    yolov10_path = os.path.join('..', 'yolov10')
    sys.path.append(yolov10_path)
    os.system('pip install huggingface_hub -i https://mirrors.cloud.tencent.com/pypi/simple')
    os.system(
        f'pip install -r {os.path.join(yolov10_path, "requirements.txt")} -i https://mirrors.cloud.tencent.com/pypi/simple')
    os.system(f'pip install -e {yolov10_path} -i https://mirrors.cloud.tencent.com/pypi/simple')
    from ultralytics import YOLOv10 as YOLO

model_file_path = os.path.join('..', 'model', 'pp_table_net.pt')
model = YOLO(model_file_path)

csv.field_size_limit(2147483647)

REAL_TABLE_WIDTH_M = 1.525
REAL_TABLE_LENGTH_M = 2.74
REAL_TABLE_HEIGHT_M = 0.76 + 0.1525
REAL_TABLE_DIAGONAL_M = (REAL_TABLE_WIDTH_M ** 2 + REAL_TABLE_LENGTH_M ** 2) ** 0.5
NOISE_THRESHOLD = 0.0006
yolo_work = False
DEBUG = True


PROGRESS_FILE = 'progress.json'



def save_progress(progress_data):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress_data, f)

def calculate_calories_burned(met, weight_kg, duration_minutes):
    calories_burned_per_minute = (met * weight_kg * 3.5) / 200
    total_calories_burned = calories_burned_per_minute * duration_minutes
    return total_calories_burned


def calculate_calories_burned_per_hour(calories_burned, total_time_minutes):
    if total_time_minutes == 0:
        return 0, "Entertainment"

    calories_burned_per_hour = (calories_burned / total_time_minutes) * 60

    if calories_burned_per_hour < 300:
        intensity = "Entertainment"
    elif 300 <= calories_burned_per_hour <= 400:
        intensity = "Moderate"
    else:
        intensity = "Competition"

    return calories_burned_per_hour, intensity


def estimate_met(average_speed, steps_count, swings_count):
    base_met = 3
    speed_factor = average_speed / 3.0
    steps_factor = steps_count / 1000
    swings_factor = swings_count / 100

    estimated_met = base_met + speed_factor + steps_factor + swings_factor
    return min(estimated_met, 12)





def calculate_chessboard_data(frame,
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
    vertical_line_2_phys = horizontal_end_point_phys - np.array([0, 0, - 76 * error_ratio], dtype=np.float32)

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


class PoseEstimation:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.templates = {"Arm": [], "Footwork": []}
        self.recording = False
        self.keypoints_data = []
        self.video_playing = False
        self.video_length = 0
        self.current_frame = 0
        self.pingpong_class = 15
        self.cap = None
        self.TEMPLATES_FILE = 'templates.csv'
        self.video_path = os.path.join('..', 'mp4', '01.mov')
        self.load_templates_from_csv()
        self.reset_variables()
        self.image_width = None
        self.image_height = None
        self.grid_rects = None
        self.show_overlay = False
        self.camera_params = None
        self.red_cross_coords = None
        self.load_chessboard_pattern_config()
        self.covered_area = set()
        self.highlight_counts = {}
        self.large_square_width = 100.0
        self.large_square_height = 75.0
        self.cap = None
        self.fps = 0
        self.delay = 0
        self.CV_CUDA_ENABLED = cv2.cuda.getCudaEnabledDeviceCount() > 0
        self.calculate_chessboard = None
        self.camera_params = self.load_camera_params()
        self.total_speeds = {'forward': 0, 'sideways': 0, 'depth': 0, 'overall': 0}
        self.count_speeds = {'forward': 0, 'sideways': 0, 'depth': 0, 'overall': 0}
        self.max_speeds = {'forward': 0, 'sideways': 0, 'depth': 0, 'overall': 0}



    def reset_variables(self):
        self.previous_midpoint = None
        self.previous_foot_points = None
        self.previous_hand_points = None
        self.previous_time = None
        self.start_time = time.time()
        self.covered_area = set()
        self.highlight_counts = {}
        self.template_match_counts = {"Arm": {}, "Footwork": {}}
        self.last_matched_templates = {"Arm": set(), "Footwork": set()}

    def load_camera_params(self):
        try:
            with open('chessboard_pattern_config.json', 'r') as f:
                config = json.load(f)
                camera_params = config.get("camera_params", {})
                mtx = np.array(camera_params["mtx"], dtype=np.float32)
                dist = np.array(camera_params["dist"], dtype=np.float32)
                rvecs = np.array(camera_params["rvecs"], dtype=np.float32)
                tvecs = np.array(camera_params["tvecs"], dtype=np.float32)
                return (mtx, dist, rvecs, tvecs)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading camera parameters: {e}")
            return None, None, None, None

    def initialize_video_capture(self, source):
        self.cap = cv2.VideoCapture(source)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        print("FPS: {}".format(self.fps))
        self.delay = int(1000 / self.fps)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def get_fps(self):
        return self.fps


    def get_total_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total_frames

    def stop_video_analysis(self):
        self.video_playing = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def is_point_in_quad(self, point, quad):
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        b1 = sign(point, quad[0], quad[1]) < 0.0
        b2 = sign(point, quad[1], quad[2]) < 0.0
        b3 = sign(point, quad[2], quad[3]) < 0.0
        b4 = sign(point, quad[3], quad[0]) < 0.0

        return ((b1 == b2) and (b2 == b3) and (b3 == b4))

    def calculate_covered_area(self, highlight_ratios):
        square_width_m = 1.0
        square_height_m = 0.75

        highlighted_squares = sum(1 for ratio in highlight_ratios.values() if ratio > 0)
        covered_area = highlighted_squares * square_width_m * square_height_m

        return covered_area

    def compare_keypoints(self, current_keypoints, template_keypoints, category, threshold=0.9):
        Arm_indices = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        Footwork_indices = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

        indices = Arm_indices if category == "Arm" else Footwork_indices

        for frame_keypoints in template_keypoints:
            if len(current_keypoints) != len(frame_keypoints):
                continue

            angles_current = []
            angles_template = []

            for idxs in [
                (11, 13, 15), (12, 14, 16),
                (23, 11, 13), (24, 12, 14),
                (13, 15, 17), (14, 16, 18),
                (23, 25, 27), (24, 26, 28),
                (26, 28, 32), (25, 27, 31),
                (28, 24, 27), (27, 23, 28)
            ]:
                if idxs[0] in indices and idxs[1] in indices and idxs[2] in indices:
                    angles_current.append(self.calculate_angle(current_keypoints[idxs[0]], current_keypoints[idxs[1]],
                                                               current_keypoints[idxs[2]]))
                    angles_template.append(self.calculate_angle(frame_keypoints[idxs[0]], frame_keypoints[idxs[1]],
                                                                frame_keypoints[idxs[2]]))

            similarity = np.mean([1 - abs(a - b) / 180 for a, b in zip(angles_current, angles_template)])
            if similarity >= threshold:
                return similarity

        return 0


    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

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
                print("Error", f"Failed to load templates from CSV: {e}")

    def convert_to_physical_coordinates(self, image_point, mtx, dist, rvec, tvec):
        image_point = np.array([image_point], dtype=np.float32)
        mtx = np.array(mtx, dtype=np.float32)  # 转换为浮点数NumPy数组
        dist = np.array(dist, dtype=np.float32)  # 转换为浮点数NumPy数组
        undistorted_point = cv2.undistortPoints(image_point, mtx, dist, P=mtx)
        rvec = np.array(rvec).reshape((3, 1))
        tvec = np.array(tvec).reshape((3, 1))
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        camera_matrix_inv = np.linalg.inv(mtx)
        uv_point = np.array([undistorted_point[0][0][0], undistorted_point[0][0][1], 1.0])
        world_point = np.dot(camera_matrix_inv, uv_point) * np.linalg.norm(tvec)
        world_point = np.dot(rotation_matrix.T, (world_point - tvec.T).T)
        return world_point.flatten()

    def convert_to_real_coordinates(self, keypoints, scaling_factor):
        real_coords = []
        for (x, y, z) in keypoints:
            X = x * scaling_factor
            Y = y * scaling_factor
            Z = z * scaling_factor
            real_coords.append((X, Y, Z))
        return real_coords

    def process_video(self, frame, pose):
        start_time = time.time()

        if self.fps == 0:  # Check if fps is not set and set it if necessary
            self.fps = 30  # Default value or calculate based on video properties


        if self.CV_CUDA_ENABLED:
            cv2.cuda.setDevice(1)
        if self.CV_CUDA_ENABLED:
            gpu_frame_start = time.time()
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            image = gpu_frame.download()
            gpu_frame_end = time.time()
            logging.info(f'GPU Frame Processing Time: {gpu_frame_end - gpu_frame_start:.4f} seconds')
        else:
            image = frame

        pose_start = time.time()
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        pose_end = time.time()
        logging.info(f'Pose Processing Time: {pose_end - pose_start:.4f} seconds')

        chessboard_start = time.time()
        if self.image_width is None or self.image_height is None:
            self.image_width = image.shape[1]
            self.image_height = image.shape[0]

        chessboard_data, output_image = self.process_chessboard(frame)
        chessboard_end = time.time()
        logging.info(f'Chessboard Processing Time: {chessboard_end - chessboard_start:.4f} seconds')

        keypoints_start = time.time()
        keypoints = []
        foot_points = []
        hand_points = []
        current_speed = {
            'forward': 0,
            'sideways': 0,
            'depth': 0,
            'overall': 0
        }

        match_results = {"Arm": {}, "Footwork": {}}  # 初始化 match_results

        if results.pose_landmarks:
            keypoints, foot_points, hand_points, current_speed = self.process_keypoints_and_speed(
                results.pose_landmarks.landmark)
            match_results = self.match_all_templates(keypoints, foot_points, hand_points)

            if self.recording:
                self.keypoints_data.append(keypoints)

            # 更新速度总和、计数器和最大值
            for k in self.total_speeds.keys():
                if current_speed[k] != 0:
                    self.total_speeds[k] += current_speed[k]
                    self.count_speeds[k] += 1
                    self.max_speeds[k] = max(self.max_speeds[k], current_speed[k])

        keypoints_end = time.time()
        logging.info(f'Keypoints and Speed Processing Time: {keypoints_end - keypoints_start:.4f} seconds')

        yolo_start = time.time()
        if yolo_work:
            detected_objects = self.detect_pingpong_table(frame, model)
            for (center_x, center_y, coord_text) in detected_objects:
                cv2.circle(output_image, (center_x, center_y), 5, (0, 255, 0), -1)
                cv2.putText(output_image, coord_text, (center_x + 10, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)
        yolo_end = time.time()
        logging.info(f'YOLO Processing Time: {yolo_end - yolo_start:.4f} seconds')

        highlight_start = time.time()
        output_image = self.process_speeds_and_highlight_ratios(keypoints, match_results, current_speed,
                                                                chessboard_data, foot_points, output_image)
        highlight_end = time.time()
        logging.info(f'Highlight Ratios Processing Time: {highlight_end - highlight_start:.4f} seconds')

        total_time = time.time() - start_time
        logging.info(f'Total Frame Processing Time: {total_time:.4f} seconds')

        logging.info(f'Processed frame: {self.current_frame}/{self.video_length}')
        return output_image


    def process_speeds_and_highlight_ratios(self, keypoints, match_results, current_speed, chessboard_data, foot_points,
                                            output_image):
        swing_count = sum(self.template_match_counts["Arm"].values())
        step_count = sum(self.template_match_counts["Footwork"].values())

        self.speeds = {
            'forward': {
                'current': current_speed['forward'],
                'max': self.max_speeds['forward'],
                'avg': self.total_speeds['forward'] / self.count_speeds['forward'] if self.count_speeds[
                    'forward'] else 0
            },
            'sideways': {
                'current': current_speed['sideways'],
                'max': self.max_speeds['sideways'],
                'avg': self.total_speeds['sideways'] / self.count_speeds['sideways'] if self.count_speeds[
                    'sideways'] else 0
            },
            'depth': {
                'current': current_speed['depth'],
                'max': self.max_speeds['depth'],
                'avg': self.total_speeds['depth'] / self.count_speeds['depth'] if self.count_speeds['depth'] else 0
            },
            'overall': {
                'current': current_speed['overall'],
                'max': self.max_speeds['overall'],
                'avg': self.total_speeds['overall'] / self.count_speeds['overall'] if self.count_speeds[
                    'overall'] else 0
            }
        }

        height_m = self.calculate_physical_height(keypoints, self.camera_params, self.image_width, self.image_height)

        skeleton_canvas = self.calculate_skeleton_image(keypoints, match_results, foot_points,
                                                        chessboard_data)


        highlight_ratios = {tuple(map(tuple, vertices)): 0 for vertices in self.grid_rects}
        for cell_points_tuple in self.highlight_counts:
            if cell_points_tuple in highlight_ratios:
                highlight_ratios[cell_points_tuple] = (self.highlight_counts[cell_points_tuple] / sum(
                    self.highlight_counts.values())) * 100


        return output_image

    def process_chessboard(self, frame):
        if self.grid_rects and self.red_cross_coords and self.camera_params and not self.calculate_chessboard:
            chessboard_data = {
                'chessboard_vertices': self.grid_rects,
                'right_top_vertex_img': self.red_cross_coords.get("right_top_vertex", (0, 0)),
                'vertical_end_point_img': self.red_cross_coords.get("vertical_end_point", (0, 0)),
                'horizontal_start_point_img': self.red_cross_coords.get("horizontal_start_point", (0, 0)),
                'horizontal_end_point_img': self.red_cross_coords.get("horizontal_end_point", (0, 0)),
                'vertical_line_1_end_img': self.red_cross_coords.get("vertical_line_1_end_img", (0, 0)),
                'vertical_line_2_end_img': self.red_cross_coords.get("vertical_line_2_end_img", (0, 0))
            }
        else:
            try:
                chessboard_data = calculate_chessboard_data(frame=frame)
                self.grid_rects = chessboard_data['chessboard_vertices']
                self.camera_params = (
                    chessboard_data['mtx'], chessboard_data['dist'], chessboard_data['rvecs'], chessboard_data['tvecs'])
                self.red_cross_coords = {
                    "right_top_vertex": chessboard_data['right_top_vertex_img'],
                    "vertical_end_point": chessboard_data['vertical_end_point_img'],
                    "horizontal_start_point": chessboard_data['horizontal_start_point_img'],
                    "horizontal_end_point": chessboard_data['horizontal_end_point_img'],
                    "vertical_line_1_end_img": chessboard_data['vertical_line_1_end_img'],
                    "vertical_line_2_end_img": chessboard_data['vertical_line_2_end_img']
                }
                chessboard_params = {
                    "small_chessboard_size": (8, 8),
                    "small_square_size": 10.0,
                    "large_square_width": 100.0,
                    "large_square_height": 75.0,
                    "vertical_offset": -15.0,
                    "num_large_squares_x": 3,
                    "num_large_squares_y": 6
                }
                self.save_chessboard_pattern(chessboard_params, self.grid_rects, self.red_cross_coords,
                                             self.camera_params)
                self.calculate_chessboard = False
            except Exception as e:
                print(f"Error in drawing large chessboard pattern: {e}")
                chessboard_data = None

        return chessboard_data, frame

    def process_keypoints_and_speed(self, landmarks):
        keypoints = [(lm.x, lm.y, lm.z) for lm in landmarks]

        left_foot_points = [(landmarks[idx].x, landmarks[idx].y, landmarks[idx].z) for idx in [29, 31] if
                            idx < len(landmarks)]
        right_foot_points = [(landmarks[idx].x, landmarks[idx].y, landmarks[idx].z) for idx in [30, 32] if
                             idx < len(landmarks)]
        foot_points = left_foot_points + right_foot_points

        left_hand_points = [(landmarks[idx].x, landmarks[idx].y) for idx in [17, 19] if idx < len(landmarks)]
        right_hand_points = [(landmarks[idx].x, landmarks[idx].y) for idx in [18, 20] if idx < len(landmarks)]
        hand_points = left_hand_points + right_hand_points

        current_speed = {
            'forward': 0,
            'sideways': 0,
            'depth': 0,
            'overall': 0
        }

        if self.previous_midpoint is not None:
            delta_time = 1.0 / self.fps
            current_midpoint = [(landmarks[23].x + landmarks[24].x) / 2,
                                (landmarks[23].y + landmarks[24].y) / 2]

            current_midpoint_phys = self.convert_to_physical_coordinates(current_midpoint, *self.camera_params)
            previous_midpoint_phys = self.convert_to_physical_coordinates(self.previous_midpoint, *self.camera_params)

            delta_distance = np.linalg.norm(current_midpoint_phys - previous_midpoint_phys)
            if delta_distance < NOISE_THRESHOLD:
                delta_distance = 0
                delta_distance_x = 0
                delta_distance_y = 0
                delta_distance_z = 0
            else:
                delta_distance_x = abs(current_midpoint_phys[0] - previous_midpoint_phys[0])
                delta_distance_y = abs(current_midpoint_phys[1] - previous_midpoint_phys[1])
                delta_distance_z = abs(current_midpoint_phys[2] - previous_midpoint_phys[2])

            current_speed['overall'] = delta_distance / delta_time
            current_speed['forward'] = delta_distance_y / delta_time
            current_speed['sideways'] = delta_distance_x / delta_time
            current_speed['depth'] = delta_distance_z / delta_time

        self.previous_midpoint = [(landmarks[23].x + landmarks[24].x) / 2,
                                  (landmarks[23].y + landmarks[24].y) / 2]

        if hand_points:
            if self.previous_hand_points is not None:
                delta_distance = np.mean([np.linalg.norm(np.array(p1) - np.array(p2)) for p1, p2 in
                                          zip(hand_points, self.previous_hand_points)])
                if delta_distance < NOISE_THRESHOLD:
                    hand_points = self.previous_hand_points
            self.previous_hand_points = hand_points

        return keypoints, foot_points, hand_points, current_speed

    def detect_pingpong_table(self, frame, model):
        table_results = model.predict(frame)
        label_map = {
            0: 'dog', 1: 'person', 2: 'cat', 3: 'tv', 4: 'car', 5: 'meatballs', 6: 'marinara sauce',
            7: 'tomato soup', 8: 'chicken noodle soup', 9: 'french onion soup', 10: 'chicken breast',
            11: 'ribs', 12: 'pulled pork', 13: 'hamburger', 14: 'cavity', 15: 'tc', 16: 'tl', 17: 'tn'
        }
        detected_objects = []

        for result in table_results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = label_map.get(cls, 'Unknown')

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                coord_text = f'{label}({center_x}, {center_y})'
                detected_objects.append((center_x, center_y, coord_text))

                box_width = x2 - x1
                box_height = y2 - y1
                scaling_factor = REAL_TABLE_DIAGONAL_M / ((box_width ** 2 + box_height ** 2) ** 0.5)

        return detected_objects

    def match_all_templates(self, current_keypoints, foot_points, hand_points):
        match_results = {"Arm": {}, "Footwork": {}}
        current_matched_templates = {"Arm": set(), "Footwork": set()}
        for category, templates in self.templates.items():
            max_similarity = 0
            best_template_name = None
            for template in templates:
                template_name = template['name']
                template_keypoints = template['data']
                similarity = self.compare_keypoints(current_keypoints, template_keypoints, category)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_template_name = template_name

            if best_template_name:
                current_matched_templates[category].add(best_template_name)
                if best_template_name not in self.last_matched_templates[category]:
                    if best_template_name not in self.template_match_counts[category]:
                        self.template_match_counts[category][best_template_name] = 0
                    self.template_match_counts[category][best_template_name] += 1
                match_results[category][best_template_name] = max_similarity

        self.last_matched_templates = current_matched_templates
        return match_results

    def analyze_video(self, video_path):
        self.initialize_video_capture(video_path)
        self.keypoints_data = []
        self.video_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.video_playing = True
        self.start_time = time.time()
        self.frame_count = 0

        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0) as pose:
            while self.cap.isOpened() and self.video_playing:
                ret, frame = self.cap.read()
                if not ret:
                    break

                image = self.process_video(frame, pose)

                self.frame_count += 1
                elapsed_time = (time.time() - self.start_time) * 1000
                expected_time = self.frame_count * self.delay
                wait_time = int(expected_time - elapsed_time)

                if wait_time > 0:
                    time.sleep(wait_time / 1000.0)

                self.current_frame += 1
                # Update progress
                progress_percentage = (self.current_frame / self.video_length) * 100
                elapsed_time = time.time() - self.start_time
                estimated_time_remaining = elapsed_time * (self.video_length - self.current_frame) / self.current_frame
                progress_data = {
                    "progress": progress_percentage,
                    "elapsed_time": elapsed_time,
                    "estimated_time_remaining": estimated_time_remaining
                }
                save_progress(progress_data)

        self.video_playing = False
        self.cap.release()
        cv2.destroyAllWindows()

    def calculate_physical_height(self, keypoints, camera_params, image_width, image_height):
        if len(keypoints) > 0:
            mtx, dist, rvecs, tvecs = camera_params

            left_ankle = keypoints[27]
            right_ankle = keypoints[28]
            nose = keypoints[0]

            left_ankle = (left_ankle[0] * image_width, left_ankle[1] * image_height, left_ankle[2])
            right_ankle = (right_ankle[0] * image_width, right_ankle[1] * image_height, right_ankle[2])
            nose = (nose[0] * image_width, nose[1] * image_height, nose[2])

            ankle_img_point = [(left_ankle[0] + right_ankle[0]) / 2, (left_ankle[1] + right_ankle[1]) / 2]
            nose_img_point = [nose[0], nose[1]]

            ankle_phys_point = self.convert_to_physical_coordinates(ankle_img_point, mtx, dist, rvecs, tvecs)
            nose_phys_point = self.convert_to_physical_coordinates(nose_img_point, mtx, dist, rvecs, tvecs)

            height_m = np.linalg.norm(nose_phys_point - ankle_phys_point) * 2.88 / 100

            return height_m
        return 0

    def calculate_skeleton_image(self, keypoints, match_results, foot_points, chessboard_data):
        screen_height = 720
        screen_width = 1280

        skeleton_canvas = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

        highlight_ratios = {tuple(map(tuple, vertices)): 0 for vertices in chessboard_data['chessboard_vertices']}

        arm_match = any(match_results["Arm"].values())
        if arm_match:
            foot_coords = [(int(foot_point[0] * screen_width), int(foot_point[1] * screen_height)) for foot_point in
                           foot_points]
            for foot_x, foot_y in foot_coords:
                if foot_x == 0 or foot_y == 0:
                    continue
                for cell_points in chessboard_data['chessboard_vertices']:
                    normalized_points = [(int(pt[0] * screen_width), int(pt[1] * screen_height)) for pt in cell_points]
                    if self.is_point_in_quad((foot_x, foot_y), normalized_points):
                        cell_points_tuple = tuple(map(tuple, cell_points))
                        self.covered_area.add(cell_points_tuple)

                        if cell_points_tuple not in self.highlight_counts:
                            self.highlight_counts[cell_points_tuple] = 0
                        self.highlight_counts[cell_points_tuple] += 1
                        break

        total_highlights = sum(self.highlight_counts.values())
        if total_highlights > 0:
            for cell_points_tuple in highlight_ratios.keys():
                highlight_ratios[cell_points_tuple] = self.highlight_counts.get(cell_points_tuple,
                                                                                0) / total_highlights * 100

        return skeleton_canvas

    def load_chessboard_pattern_config(self):
        try:
            with open('chessboard_pattern_config.json', 'r') as f:
                config = json.load(f)
                if "grid_rects" in config:
                    self.grid_rects = [tuple(tuple(map(float, pt)) for pt in cell) for cell in config["grid_rects"]]
                if "red_cross_coords" in config:
                    self.red_cross_coords = {k: tuple(map(float, v)) for k, v in config["red_cross_coords"].items()}
                if "camera_params" in config:
                    self.camera_params = config["camera_params"]
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading chessboard pattern config: {e}")
            self.grid_rects = None
            self.red_cross_coords = None
            self.camera_params = None

    def save_chessboard_pattern(self, chessboard_params, grid_rects, red_cross_coords, camera_params):
        config = {
            "chessboard_params": chessboard_params,
            "grid_rects": [list(map(list, cell)) for cell in grid_rects],
            "red_cross_coords": {k: list(v) for k, v in red_cross_coords.items()},
            "camera_params": camera_params
        }
        with open('chessboard_pattern_config.json', 'w') as f:
            json.dump(config, f, indent=4)

