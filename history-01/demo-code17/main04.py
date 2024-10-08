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

# Ensure the module for YOLOv10 is accessible
yolov10_path = os.path.join('..', 'yolov10')
sys.path.append(yolov10_path)

# Install dependencies
os.system('pip install huggingface_hub -i https://mirrors.cloud.tencent.com/pypi/simple')
os.system(f'pip install -r {os.path.join(yolov10_path, "requirements.txt")} -i https://mirrors.cloud.tencent.com/pypi/simple')
os.system(f'pip install -e {yolov10_path} -i https://mirrors.cloud.tencent.com/pypi/simple')

from ultralytics import YOLOv10

# Load the YOLOv10 model
model_file_path = os.path.join('..', 'model', 'pp_table_net.pt')
model = YOLOv10(model_file_path)

# 增加 CSV 字段大小限制
csv.field_size_limit(sys.maxsize)

# 全局常量 295（到地毯），343（桌腿到窗口踢脚线），(棋盘到右侧边缘地毯)129， 76*25（三脚架中心点）
# Tl之间114 , Tc之间149 ， Tn 高度11.5
REAL_TABLE_WIDTH_M = 1.525  # 乒乓球台宽度，单位：米
REAL_TABLE_LENGTH_M = 2.74  # 乒乓球台长度，单位：米
REAL_TABLE_HEIGHT_M = 0.76 + 0.1525  # 乒乓球台台面高加网高，单位：米
REAL_TABLE_DIAGONAL_M = (REAL_TABLE_WIDTH_M**2 + REAL_TABLE_LENGTH_M**2) ** 0.5  # 乒乓球台对角线长度，单位：米
FPS = 30  # 假设的帧率，单位：帧每秒
NOISE_THRESHOLD = 0.03  # 噪音阈值

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

    def reset_variables(self):
        self.previous_midpoint = None
        self.previous_foot_points = None
        self.previous_hand_points = None
        self.previous_time = None
        self.speeds = []
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

    def update_data_panel(self, panel, keypoints, match_results, current_speed, max_speed, avg_speed, swing_count, step_count):
        for widget in panel.winfo_children():
            widget.destroy()

        total_matches = {category: sum(self.template_match_counts[category].values()) for category in self.template_match_counts}

        for category, templates in self.templates.items():
            frame = tk.Frame(panel)
            frame.pack(fill="both", expand=True, pady=2)
            title_text = f"{category.replace('_', ' ')}"
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

                match_info = tk.Label(match_info_frame, text=f"{template_name} {match_count} ({match_percentage:.2f}%)", font=("Arial", 30), fg=text_color)
                match_info.pack(anchor="w", side="left")

        info_text = (f"Covered Area: 2 m²\n"
                     f"Speed Current: {current_speed:.2f} m/s \nMax: {max_speed:.2f} m/s Avg: {avg_speed:.2f} m/s\n")

        combined_label = tk.Label(panel, text=info_text, font=("Arial", 30, "bold"), justify=tk.LEFT, anchor="w")
        combined_label.pack(anchor="w")

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
                    angles_current.append(self.calculate_angle(current_keypoints[idxs[0]], current_keypoints[idxs[1]], current_keypoints[idxs[2]]))
                    angles_template.append(self.calculate_angle(frame_keypoints[idxs[0]], frame_keypoints[idxs[1]], frame_keypoints[idxs[2]]))

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
        window_height = int(screen_height * 0.85)
        if window_width > 0 and window_height > 0:
            image = self.resize_image_with_aspect_ratio(image, window_width, window_height)
            image = ImageTk.PhotoImage(image=image)
            panel.config(image=image)
            panel.image = image

    def convert_to_real_coordinates(self, keypoints, scaling_factor):
        real_coords = []
        for (x, y, z) in keypoints:
            X = x * scaling_factor
            Y = y * scaling_factor
            Z = z * scaling_factor
            real_coords.append((X, Y, Z))
        return real_coords

    def calculate_movement_distance(self, prev_coords, curr_coords):
        if prev_coords is None or curr_coords is None:
            return 0.0
        distance = np.sqrt(np.sum((np.array(prev_coords) - np.array(curr_coords))**2))
        return distance

    def draw_grid(self, image, rect_points):
        image_width = image.shape[1]
        image_height = image.shape[0]

        # Adjust rect_points according to the image dimensions
        rect_points = [(int(x * image_width / self.image_width), int(y * image_height / self.image_height)) for x, y in
                       rect_points]

        # Draw the outer quadrilateral
        rect_points_np = np.array(rect_points, np.int32).reshape((-1, 1, 2))
        #cv2.polylines(image, [rect_points_np], isClosed=True, color=(0, 0, 255), thickness=2)
        for (x, y) in rect_points:
            coord_text = f"({x}, {y})"
            cv2.putText(image, coord_text, (x - 50, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        cols = 3
        rows = 6
        grid_points = []

        # Calculate the intersection points of the grid
        for row in range(rows + 1):
            for col in range(cols + 1):
                x = (rect_points[0][0] * (rows - row) * (cols - col) +
                     rect_points[1][0] * (rows - row) * col +
                     rect_points[3][0] * row * (cols - col) +
                     rect_points[2][0] * row * col) / (rows * cols)
                y = (rect_points[0][1] * (rows - row) * (cols - col) +
                     rect_points[1][1] * (rows - row) * col +
                     rect_points[3][1] * row * (cols - col) +
                     rect_points[2][1] * row * col) / (rows * cols)

                grid_points.append((int(x), int(y)))

        # Collect grid cells and draw them
        grid_cells = []
        for row in range(rows):
            for col in range(cols):
                top_left = grid_points[row * (cols + 1) + col]
                top_right = grid_points[row * (cols + 1) + (col + 1)]
                bottom_right = grid_points[(row + 1) * (cols + 1) + (col + 1)]
                bottom_left = grid_points[(row + 1) * (cols + 1) + col]

                cell_points = [top_left, top_right, bottom_right, bottom_left]
                grid_cells.append(cell_points)

                # Draw the cell on the image
                pts = np.array(cell_points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(image, [pts], isClosed=True, color=(0, 0, 255), thickness=2)  

        return grid_cells

    def is_point_in_quad(self,foot_point, quad_points):
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        b1 = sign(foot_point, quad_points[0], quad_points[1]) < 0.0
        b2 = sign(foot_point, quad_points[1], quad_points[2]) < 0.0
        b3 = sign(foot_point, quad_points[2], quad_points[3]) < 0.0
        b4 = sign(foot_point, quad_points[3], quad_points[0]) < 0.0

        return (b1 == b2) and (b2 == b3) and (b3 == b4)
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

        #camera calibration

        rect_points = [(330, 720), (733, 710), (670, 523), (405, 523)]
        rect_points = np.array(rect_points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [rect_points], isClosed=True, color=(0, 0, 255), thickness=2)

        # 在左侧视频帧中绘制固定的矩形和3x6网格
        grid_rect_points = [(500, 285), (850, 300), (1225, 500), (415, 502)]
        grid_rects = self.draw_grid(image, grid_rect_points)


        keypoints = []
        foot_points = []  # Initialize foot_points with a default value
        hand_points = []  # Initialize hand_points with a default value
        current_speed = 0  # Initialize current speed with a default value
        max_speed = max(self.speeds) if self.speeds else 0
        avg_speed = np.mean(self.speeds) if self.speeds else 0

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

            image_width_m = 2
            image_height_m = 1
            if self.previous_midpoint is not None:
                delta_time = 1.0 / FPS  # 每帧的时间间隔
                current_midpoint = [(landmarks[23].x + landmarks[24].x) / 2,
                                    (landmarks[23].y + landmarks[24].y) / 2]
                delta_distance = np.linalg.norm(np.array(current_midpoint) - np.array(self.previous_midpoint))
                delta_distance *= max(image_width_m, image_height_m)  # Convert to meters

                if delta_distance < NOISE_THRESHOLD:
                    delta_distance = 0  # Ignore noise

                current_speed = delta_distance / delta_time  # Convert to m/s
                self.speeds.append(current_speed)
                max_speed = max(self.speeds)
                avg_speed = np.mean(self.speeds)

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
            self.draw_skeleton(image, keypoints, self.mp_pose.POSE_CONNECTIONS,
                               (0, 255, 0) if any(
                                   any(match_results[category].values()) for category in match_results) else (
                               255, 255, 255))
            if self.recording:
                self.keypoints_data.append(keypoints)

            for foot_point in foot_points:
                foot_x, foot_y = int(foot_point[0] * image.shape[1]), int(foot_point[1] * image.shape[0])
                for cell_points in grid_rects:
                    if self.is_point_in_quad((foot_x, foot_y), cell_points):
                        pts = np.array(cell_points, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 255), thickness=2)  # 画四边形
                        break


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
                    cv2.circle(image, (center_x, center_y), 5, (0, 255, 0), -1)
                    cv2.putText(image, coord_text, (center_x + 10, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # 计算缩放因子
                    box_width = x2 - x1
                    box_height = y2 - y1
                    scaling_factor = REAL_TABLE_DIAGONAL_M / ((box_width ** 2 + box_height ** 2) ** 0.5)

                    # 转换坐标
                    real_coords = self.convert_to_real_coordinates(keypoints, scaling_factor)

        if self.previous_midpoint is not None:
            delta_time = 1.0 / FPS  # 每帧的时间间隔
            current_midpoint = [(landmarks[23].x + landmarks[24].x) / 2,
                                (landmarks[23].y + landmarks[24].y) / 2]
            delta_distance = np.linalg.norm(np.array(current_midpoint) - np.array(self.previous_midpoint))
            delta_distance *= max(image_width_m, image_height_m)  # Convert to meters

            if delta_distance < NOISE_THRESHOLD:
                delta_distance = 0  # Ignore noise

            current_speed = delta_distance / delta_time  # Convert to m/s
            self.speeds.append(current_speed)
            max_speed = max(self.speeds)
            avg_speed = np.mean(self.speeds)

        self.previous_midpoint = [(landmarks[23].x + landmarks[24].x) / 2,
                                              (landmarks[23].y + landmarks[24].y) / 2]

        swing_count = sum(self.template_match_counts["upper_body"].values())
        step_count = sum(self.template_match_counts["lower_body"].values())

        self.update_data_panel(data_panel, keypoints, match_results, current_speed, max_speed, avg_speed,
                               swing_count, step_count)
        self.update_skeleton_image(keypoints, match_results, foot_points)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        return image


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

    def update_skeleton_image(self, keypoints, match_results, foot_points):
        image_width = right_frame_width
        image_height = int(screen_height * 0.3)  # 调整为原高度的一半
        placeholder_image = Image.new("RGB", (image_width, image_height), (255, 255, 255))
        skeleton_image_tk = ImageTk.PhotoImage(placeholder_image)

        skeleton_canvas = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        color = (0, 255, 0) if any(any(match_results[category].values()) for category in match_results) else (
        255, 255, 255)

        margin_width = 0
        margin_height = 0
        drawing_width = image_width - 2 * margin_width
        drawing_height = image_height - 2 * margin_height

        # 进行坐标转换
        grid_rect_points = [(500, 285), (850, 300), (1225, 500), (415, 502)]
        self.draw_grid(skeleton_canvas, grid_rect_points)

        # 绘制骨架
        self.draw_skeleton(skeleton_canvas, keypoints, self.mp_pose.POSE_CONNECTIONS, color, 3)

        skeleton_pil_image = Image.fromarray(cv2.cvtColor(skeleton_canvas, cv2.COLOR_BGR2RGB))
        scale = min(drawing_width / skeleton_pil_image.width, drawing_height / skeleton_pil_image.height)
        new_size = (int(skeleton_pil_image.width * scale), int(skeleton_pil_image.height * scale))
        skeleton_pil_image = skeleton_pil_image.resize(new_size, Image.Resampling.LANCZOS)

        final_image = Image.new("RGB", (image_width, image_height), (0, 0, 0))

        #final_image.paste(skeleton_pil_image, ((image_width - new_size[0]) // 2, (image_height - new_size[1]) // 2))
        final_image.paste(skeleton_pil_image,
                          ((image_width - new_size[0]) // 2 - 100, (image_height - new_size[1]) // 2 -20))

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

class PoseApp:
    def __init__(self, root, pose_estimation):
        self.root = root
        self.pose_estimation = pose_estimation
        #start with real_time
        #self.mode = "real_time"
        #self.setup_ui()
        #self.pose_estimation.start_real_time_analysis()

        #start with match_template
        self.mode = "match_template"
        self.setup_ui()
        self.update_mode_label()
        self.pose_estimation.analyze_video()

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
        mode_label = tk.Label(self.left_frame, text="Mode: Real-time Analysis", bg="green", fg="white", font=("Arial", 16))
        mode_label.pack(side="top", fill="x")

        self.top_left_frame = ttk.Frame(self.left_frame, width=left_frame_width, height=int(screen_height * 0.85), relief="solid", borderwidth=1)
        self.top_left_frame.pack(side="top", fill="both", expand=True)

        global video_panel
        video_panel = tk.Label(self.top_left_frame)
        video_panel.pack(fill="both", expand=True)

        self.progress_frame = ttk.Frame(self.left_frame, width=left_frame_width, height=20)
        self.progress_frame.pack(side="top", fill="x")

        progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Scale(self.progress_frame, from_=0, to=100, orient="horizontal", variable=progress_var, command=self.on_progress_bar_drag)
        self.progress_bar.pack(fill="x", expand=True)
        self.progress_bar.bind("<ButtonRelease-1>", self.on_progress_bar_release)
        self.progress_bar.bind("<ButtonPress-1>", self.on_progress_bar_press)

        self.right_frame = ttk.Frame(self.root, width=right_frame_width, height=screen_height)
        self.right_frame.pack(side="right", fill="both", expand=True)

        self.top_frame = tk.Frame(self.right_frame, width=right_frame_width, height=int(screen_height * 0.3), relief="solid", borderwidth=2, bg="red")
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
        elif event.keysym == 'r':
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
                self.pose_estimation.templates[category] = [t for t in self.pose_estimation.templates[category] if t['name'] != template_name]
                self.pose_estimation.update_template_listbox(template_listbox)
                self.pose_estimation.save_templates_to_csv()
        elif event.keysym == 'o':
            if any(self.pose_estimation.templates.values()):
                self.pose_estimation.close_camera()
                self.mode = "match_template"
                self.update_mode_label()
                self.pose_estimation.reset_variables()
                self.pose_estimation.analyze_video()

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
