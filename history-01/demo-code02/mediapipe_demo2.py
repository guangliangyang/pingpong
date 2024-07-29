import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from PIL import Image, ImageTk, ImageOps
import csv
import os


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
        self.current_template = None

    def draw_landmarks_with_connections(self, image, keypoints, connections, color):
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if start_idx < len(keypoints) and end_idx < len(keypoints) and start_idx > 10 and end_idx > 10:
                start_point = (
                int(keypoints[start_idx][0] * image.shape[1]), int(keypoints[start_idx][1] * image.shape[0]))
                end_point = (int(keypoints[end_idx][0] * image.shape[1]), int(keypoints[end_idx][1] * image.shape[0]))
                cv2.line(image, start_point, end_point, color, 2)
                cv2.circle(image, start_point, 5, color, -1)
                cv2.circle(image, end_point, 5, color, -1)

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

    def update_data_panel(self, panel, keypoints):
        data_text = ""
        if keypoints:
            for idx, (x, y, z) in enumerate(keypoints):
                data_text += f"Point {idx}: x={x:.2f}, y={y:.2f}, z={z:.2f}\n"
        panel.config(text=data_text)

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

    def compare_keypoints(self, current_keypoints, template_keypoints, threshold=0.8):
        if len(current_keypoints) != len(template_keypoints):
            return False

        angles_current = []
        angles_template = []

        for idxs in [(11, 13, 15), (12, 14, 16), (23, 25, 27), (24, 26, 28), (11, 23, 25), (12, 24, 26)]:
            angles_current.append(self.calculate_angle(current_keypoints[idxs[0]], current_keypoints[idxs[1]],
                                                       current_keypoints[idxs[2]]))
            angles_template.append(self.calculate_angle(template_keypoints[idxs[0]], template_keypoints[idxs[1]],
                                                        template_keypoints[idxs[2]]))

        similarity = np.mean([1 - abs(a - b) / 180 for a, b in zip(angles_current, angles_template)])
        return similarity >= threshold

    def save_templates_to_csv(self):
        with open(self.TEMPLATES_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['name', 'data'])
            for template in self.templates:
                writer.writerow([template['name'], template['data']])

    def load_templates_from_csv(self):
        if os.path.exists(self.TEMPLATES_FILE):
            with open(self.TEMPLATES_FILE, mode='r') as file:
                reader = csv.reader(file)
                next(reader)
                for row in reader:
                    name = row[0]
                    data = eval(row[1])
                    self.templates.append({'name': name, 'data': data})

    def update_template_listbox(self, listbox):
        listbox.delete(0, tk.END)
        for template in self.templates:
            listbox.insert(tk.END, template["name"])

    def update_video_panel(self, image, panel):
        window_width = left_frame_width
        window_height = int(screen_height * 0.55)
        if window_width > 0 and window_height > 0:
            image = self.resize_image_with_aspect_ratio(image, window_width, window_height)
            image = ImageTk.PhotoImage(image=image)
            panel.config(image=image)
            panel.image = image

    def process_video(self, frame, pose, template_keypoints=None):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        keypoints = []

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            keypoints = [(lm.x, lm.y, lm.z) for lm in landmarks]
            is_match = template_keypoints and self.compare_keypoints(keypoints, template_keypoints)
            self.draw_landmarks_with_connections(image, keypoints, self.mp_pose.POSE_CONNECTIONS,
                                                 (0, 255, 0) if is_match else (255, 255, 255))
            self.draw_face_triangle(image, keypoints)
            self.draw_connection_line(image, keypoints)
            if self.recording:
                self.keypoints_data.append(keypoints)

        self.update_data_panel(data_panel, keypoints)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        return image

    def analyze_video(self, video_path, template_keypoints=None):
        cap = cv2.VideoCapture(video_path)
        self.keypoints_data = []
        self.video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.video_playing = True
        self.current_template = template_keypoints

        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened() and self.video_playing:
                ret, frame = cap.read()
                if not ret:
                    break

                self.current_frame += 1
                image = self.process_video(frame, pose, template_keypoints)
                self.update_video_panel(image, video_panel)
                root.update_idletasks()
                root.update()

        self.video_playing = False
        cap.release()
        cv2.destroyAllWindows()

    def play_template_video(self, template_data):
        self.video_playing = True
        self.video_length = len(template_data)
        self.current_frame = 0

        for keypoints in template_data:
            if not self.video_playing:
                break

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            self.draw_landmarks_with_connections(frame, keypoints, self.mp_pose.POSE_CONNECTIONS, (255, 255, 255))
            self.draw_face_triangle(frame, keypoints)
            self.draw_connection_line(frame, keypoints)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            self.update_video_panel(image, template_video_panel)
            self.current_frame += 1
            root.update_idletasks()
            root.update()

        self.video_playing = False

    def start_real_time_analysis(self):
        self.cap.release()
        self.cap = cv2.VideoCapture(0)
        self.current_template = None
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
        global screen_width, screen_height, left_frame_width, right_frame_width
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

        self.top_left_frame = ttk.Frame(self.left_frame, width=left_frame_width, height=int(screen_height * 0.7),
                                        relief="solid", borderwidth=1)
        self.top_left_frame.pack(side="top", fill="both", expand=True)

        global video_panel
        video_panel = tk.Label(self.top_left_frame)
        video_panel.pack(fill="both", expand=True)

        self.bottom_left_frame = ttk.Frame(self.left_frame, width=left_frame_width, height=int(screen_height * 0.3),
                                           relief="solid", borderwidth=1)
        self.bottom_left_frame.pack(side="bottom", fill="both", expand=True)

        global template_video_panel
        template_video_panel = tk.Label(self.bottom_left_frame)
        template_video_panel.pack(fill="both", expand=True)

        self.right_frame = ttk.Frame(self.root, width=right_frame_width, height=screen_height)
        self.right_frame.pack(side="right", fill="both", expand=True)

        self.top_frame = ttk.Frame(self.right_frame, width=right_frame_width, height=int(screen_height * 0.45),
                                   relief="solid", borderwidth=1)
        self.top_frame.pack(side="top", fill="both", expand=True)

        global data_panel
        data_panel = tk.Label(self.top_frame, text="Detection Data", justify=tk.LEFT, anchor="nw")
        data_panel.pack(side="top", fill="both", expand=True)

        self.bottom_frame = ttk.Frame(self.right_frame, width=right_frame_width, height=int(screen_height * 0.55),
                                      relief="solid", borderwidth=1)
        self.bottom_frame.pack(side="bottom", fill="both", expand=True)

        template_list_label = tk.Label(self.bottom_frame, text="Saved Templates", justify=tk.LEFT, anchor="nw")
        template_list_label.pack(side="top", fill="both", expand=True)

        global template_listbox
        template_listbox = tk.Listbox(self.bottom_frame)
        template_listbox.pack(side="top", fill="both", expand=True)

        instruction_label = tk.Label(self.bottom_frame,
                                     text="Press 'r' to start real-time analysis using the webcam.\nPress 'a' to start creating an action template.\nPress 'd' to delete a selected template.\nPress 'o' to start matching action templates in the video.",
                                     justify=tk.LEFT, anchor="nw")
        instruction_label.pack(side="bottom", fill="x", expand=False)

        self.pose_estimation.load_templates_from_csv()
        self.pose_estimation.update_template_listbox(template_listbox)

        template_listbox.bind("<<ListboxSelect>>", self.on_template_select)
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
            self.pose_estimation.analyze_video('mp4/01.mp4')
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
            if self.pose_estimation.current_template:
                self.mode = "match_template"
                self.update_mode_label()
                self.pose_estimation.analyze_video('mp4/01.mp4', self.pose_estimation.current_template)

    def on_template_select(self, event):
        selection = template_listbox.curselection()
        if selection:
            index = selection[0]
            self.pose_estimation.current_template = self.pose_estimation.templates[index]["data"]
            self.pose_estimation.play_template_video(self.pose_estimation.current_template)

    def update_frame(self):
        if not self.cap.isOpened():
            self.root.after(10, self.update_frame)
            return

        if self.mode == "real_time":
            ret, frame = self.cap.read()
            if ret:
                with self.pose_estimation.mp_pose.Pose(min_detection_confidence=0.5,
                                                       min_tracking_confidence=0.5) as pose:
                    image = self.pose_estimation.process_video(frame, pose,
                                                               template_keypoints=self.pose_estimation.current_template)
                    self.pose_estimation.update_video_panel(image, video_panel)

        self.root.after(10, self.update_frame)


if __name__ == "__main__":
    root = tk.Tk()
    pose_estimation = PoseEstimation()
    app = PoseApp(root, pose_estimation)
    root.mainloop()
