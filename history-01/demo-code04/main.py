import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import csv
import os
from moviepy.editor import VideoFileClip
from threading import Thread


class PoseEstimation:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.templates = []
        self.recording = False
        self.keypoints_data = []
        self.video_playing = False
        self.video_length = 0
        self.current_frame = 0
        self.TEMPLATES_FILE = 'templates.csv'
        self.template_match_counts = {}
        self.dragging = False
        self.last_matched_templates = set()  # Store last matched templates
        self.video_path = os.path.join('../..', 'mp4', '01.mp4')  # Define video_path as a global variable
        self.load_templates_from_csv()

    def draw_landmarks_with_connections(self, image, keypoints, connections, color, circle_radius=5):
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if start_idx < len(keypoints) and end_idx < len(keypoints) and start_idx > 10 and end_idx > 10:
                start_point = (
                    int(keypoints[start_idx][0] * image.shape[1]), int(keypoints[start_idx][1] * image.shape[0]))
                end_point = (int(keypoints[end_idx][0] * image.shape[1]), int(keypoints[end_idx][1] * image.shape[0]))
                cv2.line(image, start_point, end_point, color, 2)
                cv2.circle(image, start_point, circle_radius, color, -1)
                cv2.circle(image, end_point, circle_radius, color, -1)

    def draw_face_triangle(self, image, keypoints):
        left_eye_outer_index = 3
        right_eye_outer_index = 6
        mouth_left_index = 10
        mouth_right_index = 9

        if all(idx < len(keypoints) for idx in
               [left_eye_outer_index, right_eye_outer_index, mouth_left_index, mouth_right_index]):
            left_eye_outer_point = keypoints[left_eye_outer_index][:2]
            right_eye_outer_point = keypoints[right_eye_outer_index][:2]
            mouth_left_point = keypoints[mouth_left_index][:2]
            mouth_right_point = keypoints[mouth_right_index][:2]

            left_eye_outer_point = (
                int(left_eye_outer_point[0] * image.shape[1]), int(left_eye_outer_point[1] * image.shape[0]))
            right_eye_outer_point = (
                int(right_eye_outer_point[0] * image.shape[1]), int(right_eye_outer_point[1] * image.shape[0]))
            mouth_left_point = (int(mouth_left_point[0] * image.shape[1]), int(mouth_left_point[1] * image.shape[0]))
            mouth_right_point = (int(mouth_right_point[0] * image.shape[1]), int(mouth_right_point[1] * image.shape[0]))

            triangle_cnt = np.array([left_eye_outer_point, right_eye_outer_point, mouth_left_point, mouth_right_point],
                                    np.int32)
            triangle_cnt = triangle_cnt.reshape((-1, 1, 2))
            cv2.drawContours(image, [triangle_cnt], 0, (255, 255, 255), 2)

    def draw_connection_line(self, image, keypoints):
        mouth_left_index = 9
        mouth_right_index = 10
        left_shoulder_index = 11
        right_shoulder_index = 12

        if all(idx < len(keypoints) for idx in
               [mouth_left_index, mouth_right_index, left_shoulder_index, right_shoulder_index]):
            mouth_left_point = keypoints[mouth_left_index][:2]
            mouth_right_point = keypoints[mouth_right_index][:2]
            left_shoulder_point = keypoints[left_shoulder_index][:2]
            right_shoulder_point = keypoints[right_shoulder_index][:2]

            mouth_mid_point = (
                (mouth_left_point[0] + mouth_right_point[0]) / 2, (mouth_left_point[1] + mouth_right_point[1]) / 2)
            shoulder_mid_point = (
                (left_shoulder_point[0] + right_shoulder_point[0]) / 2,
                (left_shoulder_point[1] + right_shoulder_point[1]) / 2)

            mouth_mid_point = (int(mouth_mid_point[0] * image.shape[1]), int(mouth_mid_point[1] * image.shape[0]))
            shoulder_mid_point = (
                int(shoulder_mid_point[0] * image.shape[1]), int(shoulder_mid_point[1] * image.shape[0]))

            cv2.line(image, mouth_mid_point, shoulder_mid_point, (255, 255, 255), 2)

    def update_data_panel(self, panel, keypoints, match_results):
        for widget in panel.winfo_children():
            widget.destroy()

        for i, template in enumerate(self.templates):
            template_name = template["name"]
            match_count = self.template_match_counts.get(template_name, 0)

            combined_image = self.draw_combined_image(template["data"], keypoints, template_name, match_count)
            combined_image = ImageTk.PhotoImage(combined_image)
            template_frame = tk.Frame(panel)
            template_frame.grid(row=i // 2, column=i % 2, padx=5, pady=5, sticky="w")

            template_label = tk.Label(template_frame, text=f"{template_name} {match_count}", font=("Arial", 20),
                                      fg="green")
            template_label.pack(side="top")

            combined_label = tk.Label(template_frame, image=combined_image)
            combined_label.image = combined_image
            combined_label.pack(side="top")

    def draw_combined_image(self, template_keypoints_data, athlete_keypoints, template_name, match_count):
        frame = np.zeros((220, 200, 3), dtype=np.uint8)

        template_image = np.zeros((200, 200, 3), dtype=np.uint8)
        for keypoints in template_keypoints_data:
            self.draw_landmarks_with_connections(template_image, keypoints, self.mp_pose.POSE_CONNECTIONS,
                                                 (255, 255, 255), 3)

        athlete_image = np.zeros((200, 200, 3), dtype=np.uint8)
        self.draw_landmarks_with_connections(athlete_image, athlete_keypoints, self.mp_pose.POSE_CONNECTIONS,
                                             (0, 255, 0), 3)

        combined_image = cv2.addWeighted(template_image, 0.5, athlete_image, 0.5, 0)

        # Convert to PIL image
        combined_image_pil = Image.fromarray(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))

        return combined_image_pil

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

    def compare_keypoints(self, current_keypoints, template_keypoints, threshold=0.9):
        for frame_keypoints in template_keypoints:
            if len(current_keypoints) != len(frame_keypoints):
                continue

            angles_current = []
            angles_template = []

            for idxs in [(11, 13, 15), (12, 14, 16), (23, 25, 27), (24, 26, 28), (11, 23, 25), (12, 24, 26)]:
                angles_current.append(self.calculate_angle(current_keypoints[idxs[0]], current_keypoints[idxs[1]],
                                                           current_keypoints[idxs[2]]))
                angles_template.append(
                    self.calculate_angle(frame_keypoints[idxs[0]], frame_keypoints[idxs[1]], frame_keypoints[idxs[2]]))

            similarity = np.mean([1 - abs(a - b) / 180 for a, b in zip(angles_current, angles_template)])
            if similarity >= threshold:
                return True

        return False

    def save_templates_to_csv(self):
        with open(self.TEMPLATES_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['name', 'data'])
            for template in self.templates:
                writer.writerow([template['name'], template['data']])

    def load_templates_from_csv(self):
        self.templates.clear()  # Clear the list before loading
        if os.path.exists(self.TEMPLATES_FILE):
            try:
                with open(self.TEMPLATES_FILE, mode='r') as file:
                    reader = csv.reader(file)
                    next(reader)
                    for row in reader:
                        name = row[0]
                        data = eval(row[1])
                        self.templates.append({'name': name, 'data': data})
            except (IOError, csv.Error) as e:
                messagebox.showerror("Error", f"Failed to load templates from CSV: {e}")

    def update_template_listbox(self, listbox):
        listbox.delete(0, tk.END)  # Clear the listbox before updating
        for template in self.templates:
            listbox.insert(tk.END, template["name"])

    def update_video_panel(self, image, panel):
        window_width = left_frame_width
        window_height = int(screen_height * 0.85)
        if window_width > 0 and window_height > 0:
            image = self.resize_image_with_aspect_ratio(image, window_width, window_height)
            image = ImageTk.PhotoImage(image=image)
            panel.config(image=image)
            panel.image = image

    def process_video(self, frame, pose):
        match_results = {}  # Initialize match_results to avoid UnboundLocalError
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        keypoints = []

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            keypoints = [(lm.x, lm.y, lm.z) for lm in landmarks]
            match_results = self.match_all_templates(keypoints)
            self.draw_landmarks_with_connections(image, keypoints, self.mp_pose.POSE_CONNECTIONS,
                                                 (0, 255, 0) if any(match_results.values()) else (255, 255, 255))
            self.draw_face_triangle(image, keypoints)
            self.draw_connection_line(image, keypoints)
            if self.recording:
                self.keypoints_data.append(keypoints)

        self.update_data_panel(data_panel, keypoints, match_results)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        return image

    def match_all_templates(self, current_keypoints):
        match_results = {}
        current_matched_templates = set()
        for template in self.templates:
            template_name = template['name']
            template_keypoints = template['data']
            is_match = self.compare_keypoints(current_keypoints, template_keypoints)
            if is_match:
                current_matched_templates.add(template_name)
                if template_name not in self.last_matched_templates:
                    if template_name not in self.template_match_counts:
                        self.template_match_counts[template_name] = 0
                    self.template_match_counts[template_name] += 1
            match_results[template_name] = is_match
        self.last_matched_templates = current_matched_templates
        return match_results

    def analyze_video(self):
        def play_video_with_audio():
            clip = VideoFileClip(self.video_path)
            clip.preview()  # This method plays both video and audio

        video_thread = Thread(target=play_video_with_audio)
        video_thread.start()

        cap = cv2.VideoCapture(self.video_path)
        self.keypoints_data = []
        self.video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.video_playing = True

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

    def update_progress_bar(self):
        if self.video_length > 0:
            progress = (self.current_frame / self.video_length) * 100
            progress_var.set(progress)

    def play_template_video(self, template_data):
        self.video_playing = True
        self.video_length = len(template_data)
        self.current_frame = 0

        for keypoints in template_data:
            if not self.video_playing:
                break

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            self.draw_landmarks_with_connections(frame, keypoints, self.mp_pose.POSE_CONNECTIONS, (255, 255, 255), 3)
            self.draw_face_triangle(frame, keypoints)
            self.draw_connection_line(frame, keypoints)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            self.update_video_panel(image, video_panel)
            self.update_progress_bar()
            self.current_frame += 1
            root.update_idletasks()
            root.update()

        self.video_playing = False

    def start_real_time_analysis(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(0)
        self.template_match_counts = {template['name']: 0 for template in self.templates}
        self.video_playing = True


class PoseApp:
    def __init__(self, root, pose_estimation):
        self.root = root
        self.pose_estimation = pose_estimation
        self.cap = cv2.VideoCapture(0)
        self.mode = "real_time"
        self.setup_ui()

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

        self.top_left_frame = ttk.Frame(self.left_frame, width=left_frame_width, height=int(screen_height * 0.85),
                                        relief="solid", borderwidth=1)
        self.top_left_frame.pack(side="top", fill="both", expand=True)

        global video_panel
        video_panel = tk.Label(self.top_left_frame)
        video_panel.pack(fill="both", expand=True)

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

        self.top_frame = ttk.Frame(self.right_frame, width=right_frame_width, height=int(screen_height * 1.0),
                                   relief="solid", borderwidth=1)
        self.top_frame.pack(side="top", fill="both", expand=True)

        global data_panel
        data_panel = ttk.Frame(self.top_frame)
        data_panel.pack(side="top", fill="both", expand=True)

        self.bottom_frame = ttk.Frame(self.right_frame, width=right_frame_width, height=int(screen_height * 0.25),
                                      relief="solid", borderwidth=1)
        self.bottom_frame.pack(side="bottom", fill="both", expand=True)

        template_list_label = tk.Label(self.bottom_frame, text="Saved Templates", justify=tk.LEFT, anchor="nw",
                                       font=("Arial", 12, "bold"))
        template_list_label.pack(side="top", fill="x", pady=5)

        global template_listbox
        template_listbox = tk.Listbox(self.bottom_frame, height=3)
        template_listbox.pack(side="top", fill="both", expand=True)

        instruction_label = tk.Label(self.bottom_frame,
                                     text="Press 'r' to start real-time analysis using the webcam.\nPress 'a' to start creating an action template.\nPress 'd' to delete a selected template.\nPress 'o' to start matching action templates in the video.",
                                     justify=tk.LEFT, anchor="nw")
        instruction_label.pack(side="bottom", fill="x", expand=False)

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
            self.root.destroy()
            self.cap.release()
            cv2.destroyAllWindows()
        elif event.keysym == 'a':
            self.mode = "analyze_video"
            self.update_mode_label()
            self.pose_estimation.analyze_video()
        elif event.keysym == 'b':
            self.pose_estimation.recording = True
            print("Recording started")
        elif event.keysym == 'e':
            self.pose_estimation.recording = False
            print("Recording stopped")
            if self.pose_estimation.keypoints_data:
                template_name = simpledialog.askstring("Input Template Name", "Please enter template name:")
                if template_name:
                    self.pose_estimation.templates.append(
                        {"name": template_name, "data": self.pose_estimation.keypoints_data})
                    self.pose_estimation.keypoints_data = []
                    self.pose_estimation.update_template_listbox(template_listbox)
                    self.pose_estimation.save_templates_to_csv()
        elif event.keysym == 'r':
            self.mode = "real_time"
            self.update_mode_label()
            self.pose_estimation.start_real_time_analysis()
        elif event.keysym == 'd':
            selection = template_listbox.curselection()
            if selection:
                index = selection[0]
                del self.pose_estimation.templates[index]
                self.pose_estimation.update_template_listbox(template_listbox)
                self.pose_estimation.save_templates_to_csv()
        elif event.keysym == 'o':
            if self.pose_estimation.templates:
                self.mode = "match_template"
                self.update_mode_label()
                self.pose_estimation.analyze_video()

    def update_frame(self):
        if not self.cap.isOpened():
            self.root.after(10, self.update_frame)
            return

        if self.mode == "real_time":
            ret, frame = self.cap.read()
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


if __name__ == "__main__":
    root = tk.Tk()
    pose_estimation = PoseEstimation()
    app = PoseApp(root, pose_estimation)
    root.mainloop()

'''
todo: 
右侧只显示一个命中模版，
脚没动，动作只能一个
右侧统计模版匹配相似概率
脑袋倒三角，全部加上
动作要分类，一类动作只能命中一个，挑选相似度高的。（比如碎步、交叉步、跨步是一类，左旋、右旋、上旋算一类）
'''