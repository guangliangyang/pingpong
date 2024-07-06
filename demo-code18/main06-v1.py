import os
import sys
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from PIL import Image, ImageTk
import csv
from threading import Thread
import time
import matplotlib.pyplot as plt
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

# 增加 CSV 字段大小限制
csv.field_size_limit(sys.maxsize)

# 全局常量 295（桌腿到地毯），343（桌腿到窗口踢脚线），(棋盘到右侧边缘地毯)129， 76*25（三脚架中心点）
# Tl之间114 , Tc之间149 ， Tn 高度11.5
REAL_TABLE_WIDTH_M = 1.525  # 乒乓球台宽度，单位：米
REAL_TABLE_LENGTH_M = 2.74  # 乒乓球台长度，单位：米
REAL_TABLE_HEIGHT_M = 0.76 + 0.1525  # 乒乓球台台面高加网高，单位：米
REAL_TABLE_DIAGONAL_M = (REAL_TABLE_WIDTH_M ** 2 + REAL_TABLE_LENGTH_M ** 2) ** 0.5  # 乒乓球台对角线长度，单位：米
FPS = 30  # 假设的帧率，单位：帧每秒
NOISE_THRESHOLD = 0.03  # 噪音阈值


def calculate_chessboard_data(frame,
                              small_chessboard_size=(8, 8),
                              small_square_size=10.0,
                              large_square_width=100.0,
                              large_square_height=75.0,
                              vertical_offset=-15.0,
                              num_large_squares_x=3,
                              num_large_squares_y=6):
    """
    从视频帧中提取，计算相机标定参数，并计算大棋盘格的相关数据。

    参数:
    - frame: 视频帧
    - small_chessboard_size: 小棋盘格的内角点数量 (默认为 (8, 8))
    - small_square_size: 每个小格子的实际大小 (默认为 10.0 cm)
    - large_square_width: 每个大格子的实际宽度 (默认为 100.0 cm)
    - large_square_height: 每个大格子的实际高度 (默认为 75.0 cm)
    - vertical_offset: 原点在Y方向的偏移量 (默认为 -15.0 cm)
    - num_large_squares_x: 大棋盘格在X方向的数量 (默认为 3)
    - num_large_squares_y: 大棋盘格在Y方向的数量 (默认为 6)

    返回:
    - chessboard_data: 包含计算结果的字典
    """

    def clamp_point(point, width, height):
        """确保点在图像范围内"""
        x, y = point
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        return x, y

    # 确保 frame 非空
    if frame is None:
        raise ValueError("输入帧为空，请检查图像路径是否正确")

    height, width, _ = frame.shape

    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 找到棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, small_chessboard_size, None)

    if ret:
        # 精细化角点位置
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    else:
        raise ValueError("无法检测到棋盘格角点")

    # 定义小棋盘在物理空间中的实际坐标
    objp = np.zeros((small_chessboard_size[0] * small_chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:small_chessboard_size[0], 0:small_chessboard_size[1]].T.reshape(-1, 2)
    objp *= small_square_size

    # 计算相机标定参数
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp], [corners], gray.shape[::-1], None, None)

    # 获取棋盘左下角的点作为原点
    left_bottom_corner = corners[-1][0]  # 选择左下角的角点作为原点

    # 创建大棋盘格的物理坐标，以左下角为原点，并向下移动30cm
    chessboard_physical_points = []
    for i in range(num_large_squares_y + 1):
        for j in range(num_large_squares_x + 1):
            chessboard_physical_points.append([j * large_square_width, -i * large_square_height - vertical_offset, 0])

    chessboard_physical_points = np.array(chessboard_physical_points, dtype=np.float32)

    # 将物理坐标转换为图像坐标
    def project_points(physical_points, rvec, tvec, mtx, dist):
        image_points, _ = cv2.projectPoints(physical_points, rvec, tvec, mtx, dist)
        return image_points.reshape(-1, 2)

    chessboard_image_points_px = project_points(chessboard_physical_points, rvecs[0], tvecs[0], mtx, dist)

    # 计算每个大格子的顶点在图像中的位置
    chessboard_vertices = []
    for i in range(num_large_squares_y):
        for j in range(num_large_squares_x):
            top_left = chessboard_image_points_px[i * (num_large_squares_x + 1) + j]
            top_right = chessboard_image_points_px[i * (num_large_squares_x + 1) + j + 1]
            bottom_right = chessboard_image_points_px[(i + 1) * (num_large_squares_x + 1) + j + 1]
            bottom_left = chessboard_image_points_px[(i + 1) * (num_large_squares_x + 1) + j]
            chessboard_vertices.append([top_left, top_right, bottom_right, bottom_left])

    # 计算棋盘平面的法向量
    p1 = chessboard_physical_points[0]
    p2 = chessboard_physical_points[1]
    p3 = chessboard_physical_points[num_large_squares_x + 1]
    v1 = p2 - p1
    v2 = p3 - p1
    normal_vector = np.cross(v1, v2)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)  # 归一化法向量

    # 找到最右侧中间大格子的右上角顶点的物理坐标
    middle_right_square_idx = (num_large_squares_y // 2) * num_large_squares_x + (num_large_squares_x - 1)
    right_top_vertex_idx = (num_large_squares_x + 1) * (middle_right_square_idx // num_large_squares_x) + (
            middle_right_square_idx % num_large_squares_x) + 1
    right_top_vertex_phys = chessboard_physical_points[right_top_vertex_idx]

    # 在该顶点上绘制一条垂直于棋盘的向上直线，长度为76 cm
    vertical_end_point_phys = right_top_vertex_phys + normal_vector * 76  # 向上76 cm

    right_top_vertex_img, _ = cv2.projectPoints(right_top_vertex_phys.reshape(1, 1, 3), rvecs[0], tvecs[0], mtx, dist)
    vertical_end_point_img, _ = cv2.projectPoints(vertical_end_point_phys.reshape(1, 1, 3), rvecs[0], tvecs[0], mtx,
                                                  dist)
    right_top_vertex_img = tuple(map(int, right_top_vertex_img.reshape(2)))
    vertical_end_point_img = tuple(map(int, vertical_end_point_img.reshape(2)))

    # 绘制水平线，长度为152 cm，与棋盘的Y方向平行，垂直线的终点为中点
    horizontal_start_point_phys = vertical_end_point_phys - np.array([0, 76, 0], dtype=np.float32)
    horizontal_end_point_phys = vertical_end_point_phys + np.array([0, 76, 0], dtype=np.float32)

    horizontal_start_point_img, _ = cv2.projectPoints(horizontal_start_point_phys.reshape(1, 1, 3), rvecs[0], tvecs[0],
                                                      mtx, dist)
    horizontal_end_point_img, _ = cv2.projectPoints(horizontal_end_point_phys.reshape(1, 1, 3), rvecs[0], tvecs[0], mtx,
                                                    dist)
    horizontal_start_point_img = tuple(map(int, horizontal_start_point_img.reshape(2)))
    horizontal_end_point_img = tuple(map(int, horizontal_end_point_img.reshape(2)))

    # 计算垂直线的3D物理坐标
    vertical_line_1_phys = horizontal_start_point_phys - np.array([0, 0, -76], dtype=np.float32)
    vertical_line_2_phys = horizontal_end_point_phys - np.array([0, 0, -76], dtype=np.float32)

    vertical_line_1_end_img, _ = cv2.projectPoints(vertical_line_1_phys.reshape(1, 1, 3), rvecs[0], tvecs[0], mtx, dist)
    vertical_line_2_end_img, _ = cv2.projectPoints(vertical_line_2_phys.reshape(1, 1, 3), rvecs[0], tvecs[0], mtx, dist)
    vertical_line_1_end_img = tuple(map(int, vertical_line_1_end_img.reshape(2)))
    vertical_line_2_end_img = tuple(map(int, vertical_line_2_end_img.reshape(2)))

    height, width, _ = frame.shape

    # 归一化坐标
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



def draw_chessboard_on_frame(frame, chessboard_data, show_overlay=False, covered_area=None, highlight_ratios=None):
    output_image = frame.copy()
    height, width, _ = frame.shape

    if not show_overlay:
        return output_image

    # 如果有高亮次数，使用热力图显示
    if highlight_ratios:
        max_highlight = max(highlight_ratios.values())
        norm = mcolors.Normalize(vmin=0, vmax=max_highlight)
        cmap = plt.get_cmap('hot')

    for idx, vertices in enumerate(chessboard_data['chessboard_vertices']):
        pts = np.array([(int(pt[0] * width), int(pt[1] * height)) for pt in vertices], dtype=np.int32)

        # 绘制热力图颜色
        vertex_tuples = tuple(map(tuple, vertices))  # 将vertices转换为tuple of tuples
        if highlight_ratios and vertex_tuples in highlight_ratios:
            color = cmap(norm(highlight_ratios[vertex_tuples]))[:3]  # 获取颜色
            color = tuple(int(c * 255) for c in color)  # 转换为0-255范围的颜色
            cv2.fillPoly(output_image, [pts], color=color)

        cv2.polylines(output_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # 绘制格子编号
        center_x = int(np.mean([pt[0] for pt in pts]))
        center_y = int(np.mean([pt[1] for pt in pts]))
        cv2.putText(output_image, str(idx + 1), (center_x - 10, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 绘制高亮百分比
        if highlight_ratios and vertex_tuples in highlight_ratios:
            ratio_text = f"{highlight_ratios[vertex_tuples]:.1f}%"
            cv2.putText(output_image, ratio_text, (center_x - 10, center_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            cv2.putText(output_image, "0.0%", (center_x - 10, center_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if 'right_top_vertex_img' in chessboard_data and 'vertical_end_point_img' in chessboard_data:
        right_top_vertex_img = (int(chessboard_data['right_top_vertex_img'][0] * width),
                                int(chessboard_data['right_top_vertex_img'][1] * height))
        vertical_end_point_img = (int(chessboard_data['vertical_end_point_img'][0] * width),
                                  int(chessboard_data['vertical_end_point_img'][1] * height))
        cv2.line(output_image, right_top_vertex_img, vertical_end_point_img, (255, 255, 255), 2)

    if 'horizontal_start_point_img' in chessboard_data and 'horizontal_end_point_img' in chessboard_data:
        horizontal_start_point_img = (int(chessboard_data['horizontal_start_point_img'][0] * width),
                                      int(chessboard_data['horizontal_start_point_img'][1] * height))
        horizontal_end_point_img = (int(chessboard_data['horizontal_end_point_img'][0] * width),
                                    int(chessboard_data['horizontal_end_point_img'][1] * height))
        cv2.line(output_image, horizontal_start_point_img, horizontal_end_point_img, (255, 255, 255), 2)

    if 'vertical_line_1_end_img' in chessboard_data and 'vertical_line_2_end_img' in chessboard_data:
        vertical_line_1_end_img = (int(chessboard_data['vertical_line_1_end_img'][0] * width),
                                   int(chessboard_data['vertical_line_1_end_img'][1] * height))
        vertical_line_2_end_img = (int(chessboard_data['vertical_line_2_end_img'][0] * width),
                                   int(chessboard_data['vertical_line_2_end_img'][1] * height))
        cv2.line(output_image, horizontal_start_point_img, vertical_line_1_end_img, (255, 255, 255), 2)
        cv2.line(output_image, horizontal_end_point_img, vertical_line_2_end_img, (255, 255, 255), 2)

    return output_image





class PoseEstimation:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.templates = {"upper_body": [], "lower_body": []}
        self.recording = False
        self.keypoints_data = []
        self.video_playing = False
        self.video_length = 0
        self.current_frame = 0
        self.pingpong_class = 15
        self.cap = None
        self.TEMPLATES_FILE = 'templates.csv'
        self.template_match_counts = {"upper_body": {}, "lower_body": {}}
        self.last_matched_templates = {"upper_body": set(), "lower_body": set()}
        self.dragging = False
        self.video_path = os.path.join('..', 'mp4', '01.mov')
        self.load_templates_from_csv()
        self.reset_variables()
        # Initialize global variables for image dimensions
        self.image_width = None
        self.image_height = None
        self.grid_rects = None
        self.show_overlay = True
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
        self.speeds = {
            'forward': [],
            'sideways': [],
            'overall': []
        }
        self.start_time = time.time()

    def draw_skeleton(self, image, keypoints, connections, color, circle_radius=2):

        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if start_idx < len(keypoints) and end_idx < len(keypoints) and start_idx > 10 and end_idx > 10:
                start_point = (
                    int((keypoints[start_idx][0]) * image.shape[1]), int(keypoints[start_idx][1] * image.shape[0]))
                end_point = (int((keypoints[end_idx][0]) * image.shape[1]), int(keypoints[end_idx][1] * image.shape[0]))
                cv2.line(image, start_point, end_point, color, 2)
                cv2.circle(image, start_point, circle_radius, color, -1)
                cv2.circle(image, end_point, circle_radius, color, -1)

        # Draw face triangle
        face_indices = [3, 6, 10, 9]
        if all(idx < len(keypoints) for idx in face_indices):
            points = [keypoints[idx][:2] for idx in face_indices]
            points = [(int(x * image.shape[1]), int(y * image.shape[0])) for x, y in points]
            triangle_cnt = np.array(points, np.int32).reshape((-1, 1, 2))
            cv2.drawContours(image, [triangle_cnt], 0, color, 2)

        # Draw connection line
        connection_indices = [9, 10, 11, 12]
        if all(idx < len(keypoints) for idx in connection_indices):
            points = [keypoints[idx][:2] for idx in connection_indices]
            points = [(int(x * image.shape[1]), int(y * image.shape[0])) for x, y in points]
            mouth_mid_point = ((points[0][0] + points[1][0]) // 2, (points[0][1] + points[1][1]) // 2)
            shoulder_mid_point = ((points[2][0] + points[3][0]) // 2, (points[2][1] + points[3][1]) // 2)
            cv2.line(image, mouth_mid_point, shoulder_mid_point, color, 2)

    def update_data_panel(self, panel, keypoints, match_results, speeds, swing_count, step_count, height_m):
        for widget in panel.winfo_children():
            widget.destroy()

        total_matches = {category: sum(self.template_match_counts[category].values()) for category in
                         self.template_match_counts}

        for category, templates in self.templates.items():
            frame = tk.Frame(panel)
            frame.pack(fill="both", expand=True, pady=2)
            title_text = f"{category.replace('_', ' ').title()}"
            if category == "upper_body":
                title_text += f" (Swings Count: {swing_count})"
            elif category == "lower_body":
                title_text += f" (Steps Count: {step_count})"
            title = tk.Label(frame, text=title_text, font=("Arial", 30, "bold"))
            title.pack(anchor="w")

            for template in templates:
                template_name = template["name"]
                match_count = self.template_match_counts[category].get(template_name, 0)
                match_percentage = (match_count / total_matches[category] * 100) if total_matches[category] > 0 else 0
                match_info_frame = tk.Frame(frame)
                match_info_frame.pack(fill="x", expand=True)

                similarity = match_results[category].get(template_name, 0)
                text_color = 'green' if similarity > 0 else 'black'

                bar_length = 100
                green_length = int(match_percentage * bar_length / 100)
                bar = tk.Canvas(match_info_frame, width=bar_length, height=24, bg='gray')
                bar.pack(side="left", padx=5, pady=0)
                bar.create_rectangle(0, 0, green_length, 24, fill='green', outline='')

                match_info = tk.Label(match_info_frame, text=f"{template_name} {match_count} ({match_percentage:.2f}%)",
                                      font=("Arial", 30), fg=text_color)
                match_info.pack(anchor="w", side="left")

        # 更新速度统计面板
        self.app.update_speed_stats(speeds)

        # 添加人体身高信息
        #height_info = tk.Label(panel, text=f"Height (m): {height_m:.2f}", font=("Arial", 20, "bold"), fg='blue')
        #height_info.pack(anchor="w")

    def resize_image_with_aspect_ratio(self, image, target_width, target_height):
        original_width, original_height = image.size
        if original_width == 0 or original_height == 0:
            return Image.new("RGB", (target_width, target_height))

        ratio = min(target_width / original_width, target_height / original_height)
        new_width = max(1, int(original_width * ratio))
        new_height = max(1, int(original_height * ratio))
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        new_image = Image.new("RGB", (target_width, target_height))
        new_image.paste(image, ((target_width - new_width) // 2, (target_height - new_height) // 2))
        return new_image

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    def compare_keypoints(self, current_keypoints, template_keypoints, category, threshold=0.9):
        upper_body_indices = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        lower_body_indices = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

        indices = upper_body_indices if category == "upper_body" else lower_body_indices

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

    def save_templates_to_csv(self):
        with open(self.TEMPLATES_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['name', 'category', 'data'])
            for category, templates in self.templates.items():
                for template in templates:
                    writer.writerow([template['name'], category, template['data']])

    def load_templates_from_csv(self):
        self.templates = {"upper_body": [], "lower_body": []}
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
                messagebox.showerror("Error", f"Failed to load templates from CSV: {e}")

    def update_template_listbox(self, listbox):
        listbox.delete(0, tk.END)
        for category, templates in self.templates.items():
            for template in templates:
                listbox.insert(tk.END, f"{template['name']} ({category})")

    def update_video_panel(self, image, panel):
        window_width = left_frame_width
        window_height = int(window_width / (16 / 9))  # 16:9 aspect ratio
        if window_width > 0 and window_height > 0:
            image = self.resize_image_with_aspect_ratio(image, window_width, window_height)
            image = ImageTk.PhotoImage(image=image)
            panel.config(image=image)
            panel.image = image

    def convert_to_physical_coordinates(self, image_point, mtx, dist, rvec, tvec):
        """
        将图像坐标转换为物理坐标 (3D)

        参数:
        - image_point: 图像坐标 (x, y)
        - mtx: 相机内参矩阵
        - dist: 相机畸变系数
        - rvec: 旋转向量
        - tvec: 平移向量

        返回:
        - 物理坐标 (X, Y, Z)
        """
        image_point = np.array([image_point], dtype=np.float32)
        undistorted_point = cv2.undistortPoints(image_point, mtx, dist, P=mtx)
        # 使用反投影来获得物理坐标
        rvec = np.array(rvec).reshape((3, 1))  # 确保 rvec 是一个 NumPy 数组并且是正确的形状
        tvec = np.array(tvec).reshape((3, 1))  # 确保 tvec 是一个 NumPy 数组并且是正确的形状
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        camera_matrix_inv = np.linalg.inv(mtx)
        uv_point = np.array([undistorted_point[0][0][0], undistorted_point[0][0][1], 1.0])
        world_point = np.dot(camera_matrix_inv, uv_point) * np.linalg.norm(tvec)
        world_point = np.dot(rotation_matrix.T, (world_point - tvec.T).T)
        return world_point.flatten()

    def process_video(self, frame, pose):
        match_results = {"upper_body": {}, "lower_body": {}}
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

        # 加载棋盘配置来绘制棋盘
        if self.grid_rects and self.red_cross_coords and self.camera_params and not self.app.calculate_chessboard:
            chessboard_data = {
                'chessboard_vertices': self.grid_rects,
                'right_top_vertex_img': self.red_cross_coords.get("right_top_vertex", (0, 0)),
                'vertical_end_point_img': self.red_cross_coords.get("vertical_end_point", (0, 0)),
                'horizontal_start_point_img': self.red_cross_coords.get("horizontal_start_point", (0, 0)),
                'horizontal_end_point_img': self.red_cross_coords.get("horizontal_end_point", (0, 0)),
                'vertical_line_1_end_img': self.red_cross_coords.get("vertical_line_1_end_img", (0, 0)),
                'vertical_line_2_end_img': self.red_cross_coords.get("vertical_line_2_end_img", (0, 0))

            }
            output_image = draw_chessboard_on_frame(frame, chessboard_data, show_overlay=self.show_overlay)
        else:
            try:
                chessboard_data = calculate_chessboard_data(frame=frame)
                output_image = draw_chessboard_on_frame(frame=frame, chessboard_data=chessboard_data,
                                                        show_overlay=self.show_overlay)
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
                self.app.calculate_chessboard = False  # 重置控制变量
            except Exception as e:
                print(f"Error in drawing large chessboard pattern: {e}")
                chessboard_data = None

        if chessboard_data:
            output_image = draw_chessboard_on_frame(frame, chessboard_data, show_overlay=self.show_overlay)
        else:
            output_image = frame.copy()

        keypoints = []
        foot_points = []  # Initialize foot_points with a default value
        hand_points = []  # Initialize hand_points with a default value
        current_speed = {
            'forward': 0,
            'sideways': 0,
            'overall': 0
        }

        real_coords = []  # Initialize real_coords
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            keypoints = [(lm.x, lm.y, lm.z) for lm in landmarks]

            # 提取脚的关键点
            left_foot_points = [(landmarks[idx].x, landmarks[idx].y) for idx in [29, 31] if idx < len(landmarks)]
            right_foot_points = [(landmarks[idx].x, landmarks[idx].y) for idx in [30, 32] if idx < len(landmarks)]
            foot_points = left_foot_points + right_foot_points

            # 提取手的关键点
            left_hand_points = [(landmarks[idx].x, landmarks[idx].y) for idx in [17, 19] if idx < len(landmarks)]
            right_hand_points = [(landmarks[idx].x, landmarks[idx].y) for idx in [18, 20] if idx < len(landmarks)]
            hand_points = left_hand_points + right_hand_points

            if self.previous_midpoint is not None:
                delta_time = 1.0 / FPS  # 每帧的时间间隔
                current_midpoint = [(landmarks[23].x + landmarks[24].x) / 2,
                                    (landmarks[23].y + landmarks[24].y) / 2]

                # Convert to physical coordinates
                current_midpoint_phys = self.convert_to_physical_coordinates(current_midpoint, *self.camera_params)
                previous_midpoint_phys = self.convert_to_physical_coordinates(self.previous_midpoint,
                                                                              *self.camera_params)

                delta_distance = np.linalg.norm(current_midpoint_phys - previous_midpoint_phys)
                if delta_distance < NOISE_THRESHOLD:
                    delta_distance = 0  # Ignore noise

                delta_distance_x = abs(current_midpoint_phys[0] - previous_midpoint_phys[0])
                delta_distance_y = abs(current_midpoint_phys[1] - previous_midpoint_phys[1])

                current_speed['overall'] = delta_distance / delta_time
                current_speed['forward'] = delta_distance_y / delta_time
                current_speed['sideways'] = delta_distance_x / delta_time

                self.speeds['overall'].append(current_speed['overall'])
                self.speeds['forward'].append(current_speed['forward'])
                self.speeds['sideways'].append(current_speed['sideways'])

            self.previous_midpoint = [(landmarks[23].x + landmarks[24].x) / 2,
                                      (landmarks[23].y + landmarks[24].y) / 2]

            if hand_points:
                if self.previous_hand_points is not None:
                    delta_distance = np.mean([np.linalg.norm(np.array(p1) - np.array(p2)) for p1, p2 in
                                              zip(hand_points, self.previous_hand_points)])
                    if delta_distance < NOISE_THRESHOLD:
                        hand_points = self.previous_hand_points  # Ignore noise
                self.previous_hand_points = hand_points

            match_results = self.match_all_templates(keypoints, foot_points, hand_points)
            if self.show_overlay:
                self.draw_skeleton(output_image, keypoints, self.mp_pose.POSE_CONNECTIONS,
                                   (0, 255, 0) if any(
                                       any(match_results[category].values()) for category in match_results) else (
                                       255, 255, 255))
                for foot_point in foot_points:
                    foot_x, foot_y = foot_point[0], foot_point[1]
                    for cell_points in self.grid_rects:
                        if self.is_point_in_quad((foot_x, foot_y), cell_points):
                            pts = np.array(cell_points, dtype=np.int32).reshape((-1, 1, 2))
                            cv2.polylines(output_image, [pts], isClosed=True, color=(0, 255, 255), thickness=2)  # 画四边形
                            break

            if self.recording:
                self.keypoints_data.append(keypoints)

        # YOLOv10 inference for ping pong table detection
        yolo_work = False
        if yolo_work:
            table_results = model.predict(frame)
            label_map = {
                0: 'dog', 1: 'person', 2: 'cat', 3: 'tv', 4: 'car', 5: 'meatballs', 6: 'marinara sauce',
                7: 'tomato soup', 8: 'chicken noodle soup', 9: 'french onion soup', 10: 'chicken breast',
                11: 'ribs', 12: 'pulled pork', 13: 'hamburger', 14: 'cavity', 15: 'tc', 16: 'tl', 17: 'tn'
            }
            for result in table_results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = label_map.get(cls, 'Unknown')

                    # 绘制中心点
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    coord_text = f'{label}({center_x}, {center_y})'
                    cv2.circle(output_image, (center_x, center_y), 5, (0, 255, 0), -1)
                    cv2.putText(output_image, coord_text, (center_x + 10, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 2)

                    # 计算缩放因子
                    box_width = x2 - x1
                    box_height = y2 - y1
                    scaling_factor = REAL_TABLE_DIAGONAL_M / ((box_width ** 2 + box_height ** 2) ** 0.5)

                    # 转换坐标
                    real_coords = self.convert_to_real_coordinates(keypoints, scaling_factor)

        swing_count = sum(self.template_match_counts["upper_body"].values())
        step_count = sum(self.template_match_counts["lower_body"].values())

        speeds = {
            'forward': {
                'current': current_speed['forward'],
                'max': max(self.speeds['forward']) if self.speeds['forward'] else 0,
                'avg': np.mean(self.speeds['forward']) if self.speeds['forward'] else 0
            },
            'sideways': {
                'current': current_speed['sideways'],
                'max': max(self.speeds['sideways']) if self.speeds['sideways'] else 0,
                'avg': np.mean(self.speeds['sideways']) if self.speeds['sideways'] else 0
            },
            'overall': {
                'current': current_speed['overall'],
                'max': max(self.speeds['overall']) if self.speeds['overall'] else 0,
                'avg': np.mean(self.speeds['overall']) if self.speeds['overall'] else 0
            }
        }

        height_m = self.calculate_physical_height(keypoints, self.camera_params, self.image_width, self.image_height)

        self.update_data_panel(data_panel, keypoints, match_results, speeds, swing_count, step_count, height_m)

        self.update_skeleton_image(keypoints, match_results, foot_points, chessboard_data)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        output_image = Image.fromarray(output_image)
        return output_image

    def match_all_templates(self, current_keypoints, foot_points, hand_points):
        match_results = {"upper_body": {}, "lower_body": {}}
        current_matched_templates = {"upper_body": set(), "lower_body": set()}
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

    def analyze_video(self):
        def play_video_with_audio():
            cap = cv2.VideoCapture(self.video_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                self.frame_to_show = frame
                self.new_frame = True
                if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
                    break
            cap.release()
            cv2.destroyAllWindows()

        self.new_frame = False
        self.frame_to_show = None

        video_thread = Thread(target=play_video_with_audio)
        video_thread.start()

        cap = cv2.VideoCapture(self.video_path)
        self.keypoints_data = []
        self.video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.video_playing = True
        self.start_time = time.time()

        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened() and self.video_playing:
                if not self.dragging:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                ret, frame = cap.read()
                if not ret:
                    break

                if not self.dragging:
                    self.current_frame += 1
                image = self.process_video(frame, pose)
                self.update_video_panel(image, video_panel)
                self.update_progress_bar()
                root.update_idletasks()
                root.update()

        self.video_playing = False
        cap.release()
        cv2.destroyAllWindows()

    def is_point_in_quad(self, point, quad):
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        b1 = sign(point, quad[0], quad[1]) < 0.0
        b2 = sign(point, quad[1], quad[2]) < 0.0
        b3 = sign(point, quad[2], quad[3]) < 0.0
        b4 = sign(point, quad[3], quad[0]) < 0.0

        return ((b1 == b2) and (b2 == b3) and (b3 == b4))

    def calculate_physical_height(self,keypoints, camera_params, image_width, image_height):
        """
        计算物理身高
        :param keypoints: 关键点
        :param camera_params: 相机参数（包括内参矩阵和畸变系数）
        :return: 物理身高（米）
        """
        if len(keypoints) > 0:
            mtx, dist, rvecs, tvecs = camera_params

            left_ankle = keypoints[27]
            right_ankle = keypoints[28]
            nose = keypoints[0]

            # 反归一化
            left_ankle = (left_ankle[0] * image_width, left_ankle[1] * image_height, left_ankle[2])
            right_ankle = (right_ankle[0] * image_width, right_ankle[1] * image_height, right_ankle[2])
            nose = (nose[0] * image_width, nose[1] * image_height, nose[2])

            # 使用脚踝和头顶（鼻子）之间的距离计算身高
            ankle_img_point = [(left_ankle[0] + right_ankle[0]) / 2, (left_ankle[1] + right_ankle[1]) / 2]
            nose_img_point = [nose[0], nose[1]]

            ankle_phys_point = self.convert_to_physical_coordinates(ankle_img_point, mtx, dist, rvecs, tvecs)
            nose_phys_point = self.convert_to_physical_coordinates(nose_img_point, mtx, dist, rvecs, tvecs)

            height_m = np.linalg.norm(nose_phys_point - ankle_phys_point) / 100  # 转换为米
            return height_m
        return 0

    def update_skeleton_image(self, keypoints, match_results, foot_points, chessboard_data):
        image_width = right_frame_width
        image_height = int(screen_height * 0.3)
        placeholder_image = Image.new("RGB", (image_width, image_height), (255, 255, 255))
        skeleton_image_tk = ImageTk.PhotoImage(placeholder_image)

        skeleton_canvas = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        color = (0, 255, 0) if any(any(match_results[category].values()) for category in match_results) else (
        255, 255, 255)

        # 初始化highlight_ratios
        highlight_ratios = {tuple(map(tuple, vertices)): 0 for vertices in chessboard_data['chessboard_vertices']}

        # 仅在upper_body模板命中时高亮脚踩到的格子
        if any(match_results["upper_body"].values()) or 1==1:
            for foot_point in foot_points:
                foot_x, foot_y = int(foot_point[0] * image_width), int(foot_point[1] * image_height)
                for cell_points in chessboard_data['chessboard_vertices']:
                    normalized_points = [(int(pt[0] * image_width), int(pt[1] * image_height)) for pt in cell_points]
                    if self.is_point_in_quad((foot_x, foot_y), normalized_points):
                        pts = np.array(normalized_points, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(skeleton_canvas, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
                        cell_points_tuple = tuple(map(tuple, cell_points))  # 将该格子添加到covered_area中
                        self.covered_area.add(cell_points_tuple)

                        # 更新高亮次数统计
                        if cell_points_tuple not in self.highlight_counts:
                            self.highlight_counts[cell_points_tuple] = 0
                        self.highlight_counts[cell_points_tuple] += 1
                        break

        # 计算高亮次数总和
        total_highlights = sum(self.highlight_counts.values())
        if total_highlights > 0:
            for cell_points_tuple in highlight_ratios.keys():
                highlight_ratios[cell_points_tuple] = self.highlight_counts.get(cell_points_tuple,
                                                                                0) / total_highlights * 100

        # 绘制棋盘格
        skeleton_canvas = draw_chessboard_on_frame(skeleton_canvas, chessboard_data, show_overlay=True,
                                                   covered_area=self.covered_area, highlight_ratios=highlight_ratios)

        # 绘制骨架
        self.draw_skeleton(skeleton_canvas, keypoints, self.mp_pose.POSE_CONNECTIONS, color, 3)

        skeleton_pil_image = Image.fromarray(cv2.cvtColor(skeleton_canvas, cv2.COLOR_BGR2RGB))
        scale = min(image_width / skeleton_pil_image.width, image_height / skeleton_pil_image.height)
        new_size = (int(skeleton_pil_image.width * scale), int(skeleton_pil_image.height * scale))
        skeleton_pil_image = skeleton_pil_image.resize(new_size, Image.Resampling.LANCZOS)

        final_image = Image.new("RGB", (image_width, image_height), (0, 0, 0))
        final_image.paste(skeleton_pil_image, ((image_width - new_size[0]) // 2, (image_height - new_size[1]) // 2))

        skeleton_image_tk = ImageTk.PhotoImage(final_image)

        skeleton_image_label.config(image=skeleton_image_tk)
        skeleton_image_label.image = skeleton_image_tk


    def update_progress_bar(self):
        if self.video_length > 0:
            progress = (self.current_frame / self.video_length) * 100
            progress_var.set(progress)

    def start_real_time_analysis(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(0)
        self.template_match_counts = {"upper_body": {}, "lower_body": {}}
        self.video_playing = True
        self.reset_variables()

    def close_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def save_chessboard_pattern(self, chessboard_params, grid_rects, red_cross_coords, camera_params):
        config = {
            "chessboard_params": chessboard_params,
            "grid_rects": grid_rects,
            "red_cross_coords": red_cross_coords,
            "camera_params": {
                "mtx": camera_params[0].tolist(),
                "dist": camera_params[1].tolist(),
                "rvecs": [rvec.tolist() for rvec in camera_params[2]],
                "tvecs": [tvec.tolist() for tvec in camera_params[3]]
            }
        }
        with open("chessboard_pattern_config.json", "w") as f:
            json.dump(config, f, indent=4)

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
    def __init__(self, root, pose_estimation):
        self.root = root
        self.pose_estimation = pose_estimation
        self.calculate_chessboard = False  # 初始化控制变量
        self.pose_estimation.app = self  # 将 PoseApp 实例赋值给 PoseEstimation 的 app 属性
        # start with real_time
        # self.mode = "real_time"
        # self.setup_ui()
        # self.pose_estimation.start_real_time_analysis()

        # start with match_template
        self.mode = "match_template"
        self.setup_ui()
        self.update_mode_label()
        self.pose_estimation.analyze_video()

    def on_template_label_double_click(self, event):
        label = event.widget
        template_name = label.cget("text").split()[0]
        for category, templates in self.pose_estimation.templates.items():
            if any(t['name'] == template_name for t in templates):
                confirm = messagebox.askyesno("Confirm Deletion",
                                              f"Do you want to delete the template '{template_name}'?")
                if confirm:
                    self.pose_estimation.templates[category] = [t for t in templates if t['name'] != template_name]
                    self.pose_estimation.update_template_listbox(template_listbox)
                    self.pose_estimation.save_templates_to_csv()
                break

    def setup_ui(self):
        self.root.title("Pose Estimation and Analysis")
        self.root.state('zoomed')
        global screen_width, screen_height, left_frame_width, right_frame_width, progress_var
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        left_frame_width = int(screen_width * 0.68)
        right_frame_width = screen_width - left_frame_width

        self.left_frame = ttk.Frame(self.root, width=left_frame_width, height=screen_height)
        self.left_frame.pack(side="left", fill="y", expand=False)

        global mode_label
        mode_label = tk.Label(self.left_frame, text="Mode: Real-time Analysis", bg="green", fg="white",
                              font=("Arial", 16))
        mode_label.pack(side="top", fill="x")

        # 调整视频面板位置和大小
        video_aspect_ratio = 16 / 9  # 视频的宽高比，例如16:9
        video_panel_width = left_frame_width
        video_panel_height = int(video_panel_width / video_aspect_ratio)
        self.video_panel_frame = ttk.Frame(self.left_frame, width=video_panel_width, height=video_panel_height)
        self.video_panel_frame.pack(side="top", fill="both", expand=True)

        global video_panel
        video_panel = tk.Label(self.video_panel_frame)
        video_panel.pack(fill="both", expand=True)

        # 添加速度统计面板
        self.speed_stats_panel = ttk.Frame(self.left_frame, width=left_frame_width, height=int(screen_height * 0.15))
        self.speed_stats_panel.pack(side="top", fill="both", expand=True)
        self.speed_stats_label = tk.Label(self.speed_stats_panel, text="", font=("Arial", 16))
        self.speed_stats_label.pack(fill="both", expand=True)

        self.progress_frame = ttk.Frame(self.left_frame, width=left_frame_width, height=20)
        self.progress_frame.pack(side="top", fill="x")

        progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Scale(self.progress_frame, from_=0, to=100, orient="horizontal", variable=progress_var,
                                      command=self.on_progress_bar_drag)
        self.progress_bar.pack(fill="x", expand=True)
        self.progress_bar.bind("<ButtonRelease-1>", self.on_progress_bar_release)
        self.progress_bar.bind("<ButtonPress-1>", self.on_progress_bar_press)

        self.right_frame = ttk.Frame(self.root, width=right_frame_width, height=screen_height)
        self.right_frame.pack(side="right", fill="both", expand=True)

        self.top_frame = tk.Frame(self.right_frame, width=right_frame_width, height=int(screen_height * 0.3),
                                  relief="solid", borderwidth=2, bg="red")
        self.top_frame.pack(side="top", fill="both", expand=True)

        global skeleton_image_label
        skeleton_image_label = tk.Label(self.top_frame, bg="black")
        skeleton_image_label.pack(side="top", fill="both", expand=True)

        global data_panel
        data_panel = ttk.Frame(self.right_frame)
        data_panel.pack(side="top", fill="both", expand=True)

        global template_listbox
        template_listbox = tk.Listbox(self.right_frame, height=3)

        self.pose_estimation.load_templates_from_csv()
        self.pose_estimation.update_template_listbox(template_listbox)

        self.root.bind("<KeyPress>", self.on_key_press)
        self.root.after(10, self.update_frame)

    def update_mode_label(self):
        mode_text = {
            "real_time": "Mode: Real-time Analysis",
            "analyze_video": "Mode: Analyze Video",
            "match_template": "Mode: Match Template"
        }
        mode_label.config(text=mode_text.get(self.mode, "Mode: Unknown"))

    def on_key_press(self, event):
        if event.keysym == 'Escape':
            self.pose_estimation.close_camera()
            self.root.destroy()
            cv2.destroyAllWindows()
        elif event.keysym == 'a':
            self.pose_estimation.close_camera()
            self.mode = "analyze_video"
            self.update_mode_label()
            self.pose_estimation.reset_variables()
            self.pose_estimation.analyze_video()
        elif event.keysym == 'b':
            self.pose_estimation.recording = True
            print("Recording started")
        elif event.keysym == 'e':
            self.pose_estimation.recording = False
            print("Recording stopped")
            if self.pose_estimation.keypoints_data:
                input_dialog = TemplateInputDialog(self.root)
                self.root.wait_window(input_dialog.dialog)
                if input_dialog.template_name and input_dialog.category:
                    self.pose_estimation.templates[input_dialog.category].append(
                        {"name": input_dialog.template_name, "data": self.pose_estimation.keypoints_data})
                    self.pose_estimation.keypoints_data = []
                    self.pose_estimation.update_template_listbox(template_listbox)
                    self.pose_estimation.save_templates_to_csv()
        elif event.keysym == 'F5':
            self.pose_estimation.close_camera()
            self.mode = "real_time"
            self.update_mode_label()
            self.pose_estimation.reset_variables()
            self.pose_estimation.start_real_time_analysis()
        elif event.keysym == 'd':
            selection = template_listbox.curselection()
            if selection:
                index = selection[0]
                selected_template = template_listbox.get(index)
                template_name, category = selected_template.rsplit(' (', 1)
                category = category.rstrip(')')
                self.pose_estimation.templates[category] = [t for t in self.pose_estimation.templates[category] if
                                                            t['name'] != template_name]
                self.pose_estimation.update_template_listbox(template_listbox)
                self.pose_estimation.save_templates_to_csv()
        elif event.keysym == 'F6':
            if any(self.pose_estimation.templates.values()):
                self.pose_estimation.close_camera()
                self.mode = "match_template"
                self.update_mode_label()
                self.pose_estimation.reset_variables()
                self.pose_estimation.analyze_video()
        elif event.keysym == 'F1':
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
        elif event.keysym == 'F2':
            self.calculate_chessboard = True

        elif event.keysym == 'F3':
            self.pose_estimation.show_overlay = not self.pose_estimation.show_overlay


    def update_frame(self):
        if not self.pose_estimation.cap or not self.pose_estimation.cap.isOpened():
            self.root.after(10, self.update_frame)
            return

        if self.mode == "real_time":
            ret, frame = self.pose_estimation.cap.read()
            if ret:
                with self.pose_estimation.mp_pose.Pose(min_detection_confidence=0.5,
                                                       min_tracking_confidence=0.5) as pose:
                    image = self.pose_estimation.process_video(frame, pose)
                    self.pose_estimation.update_video_panel(image, video_panel)

        self.root.after(10, self.update_frame)

    def on_progress_bar_drag(self, value):
        self.pose_estimation.dragging = True
        if self.pose_estimation.video_length > 0:
            frame_number = int((float(value) / 100) * self.pose_estimation.video_length)
            self.pose_estimation.current_frame = frame_number
            self.update_video_to_frame(frame_number)

    def on_progress_bar_press(self, event):
        self.pose_estimation.dragging = True

    def on_progress_bar_release(self, event):
        self.pose_estimation.dragging = False

    def update_video_to_frame(self, frame_number):
        cap = cv2.VideoCapture(self.pose_estimation.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            with self.pose_estimation.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                image = self.pose_estimation.process_video(frame, pose)
                self.pose_estimation.update_video_panel(image, video_panel)
        cap.release()

    def update_speed_stats(self, speeds):
        def mps_to_kph(speed_mps):
            return speed_mps * 3.6

        # 计算覆盖面积（平方米）
        covered_area_size = len(self.pose_estimation.covered_area) * (self.pose_estimation.large_square_width / 100) * (
                    self.pose_estimation.large_square_height / 100)

        # 初始化highlight_ratios
        highlight_ratios = {tuple(map(tuple, vertices)): 0 for vertices in self.pose_estimation.grid_rects}

        # 计算高亮次数总和
        total_highlights = sum(self.pose_estimation.highlight_counts.values())
        if total_highlights > 0:
            for cell_points_tuple in highlight_ratios.keys():
                highlight_ratios[cell_points_tuple] = self.pose_estimation.highlight_counts.get(cell_points_tuple,
                                                                                                0) / total_highlights * 100

        highlight_info = "\n".join([f"Cell {i + 1}: {ratio:.2f}%" for i, ratio in enumerate(highlight_ratios.values())])

        info_text = (
            f"Forward Speed:\nCurrent: {mps_to_kph(speeds['forward']['current']):.2f} km/h, Max: {mps_to_kph(speeds['forward']['max']):.2f} km/h, Avg: {mps_to_kph(speeds['forward']['avg']):.2f} km/h\n"
            f"Sideways Speed:\nCurrent: {mps_to_kph(speeds['sideways']['current']):.2f} km/h, Max: {mps_to_kph(speeds['sideways']['max']):.2f} km/h, Avg: {mps_to_kph(speeds['sideways']['avg']):.2f} km/h\n"
            f"Overall Speed:\nCurrent: {mps_to_kph(speeds['overall']['current']):.2f} km/h, Max: {mps_to_kph(speeds['overall']['max']):.2f} km/h, Avg: {mps_to_kph(speeds['overall']['avg']):.2f} km/h\n"
            f"Covered Area Size: {covered_area_size:.2f} m²\n"
            f"Highlight Ratios:\n{highlight_info}")
        self.speed_stats_label.config(text=info_text)


class TemplateInputDialog:
    def __init__(self, parent):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Input Template Information")
        self.dialog.geometry("300x225")
        self.dialog.grab_set()

        screen_width = self.dialog.winfo_screenwidth()
        screen_height = self.dialog.winfo_screenheight()
        x = (screen_width - 300) // 2
        y = (screen_height - 225) // 2
        self.dialog.geometry(f"+{x}+{y}")

        tk.Label(self.dialog, text="Template Name:").pack(pady=5)
        self.template_name_entry = tk.Entry(self.dialog)
        self.template_name_entry.pack(pady=5)

        tk.Label(self.dialog, text="Template Category:").pack(pady=5)
        self.category_var = tk.StringVar()
        self.category_combobox = ttk.Combobox(self.dialog, textvariable=self.category_var)
        self.category_combobox['values'] = ("upper_body", "lower_body")
        self.category_combobox.pack(pady=5)

        self.save_button = tk.Button(self.dialog, text="Save", command=self.save)
        self.save_button.pack(pady=5)

        self.template_name = None
        self.category = None

    def save(self):
        self.template_name = self.template_name_entry.get()
        self.category = self.category_var.get()
        self.dialog.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    pose_estimation = PoseEstimation()
    app = PoseApp(root, pose_estimation)
    root.mainloop()
