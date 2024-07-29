import os
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import csv
import time
import json

# 增加 CSV 字段大小限制
csv.field_size_limit(2147483647)

class PoseAnnotator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.load_config()
        self.reset_variables()
        self.cap = None  # 初始化 cap 属性

    def load_config(self):
        with open('config.json', 'r') as f:
            self.config = json.load(f)

    def reset_variables(self):
        self.previous_midpoint = None
        self.previous_foot_points = None
        self.previous_hand_points = None
        self.previous_time = 0
        self.start_time = 0
        self.annotations = []
        self.recording = False
        self.keypoints_data = []
        self.video_playing = False
        self.video_length = 0
        self.current_frame = 0
        self.video_path = ''
        self.start_frame = 0
        self.end_frame = 0
        self.dragging = False
        self.paused = False
        self.pause_frame = 0

    def stop_video_playback(self):
        self.video_playing = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def draw_skeleton(self, image, keypoints, connections, color, circle_radius=2):
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                start_point = (
                    int((keypoints[start_idx][0]) * image.shape[1]), int(keypoints[start_idx][1] * image.shape[0]))
                end_point = (int((keypoints[end_idx][0]) * image.shape[1]), int(keypoints[end_idx][1] * image.shape[0]))
                cv2.line(image, start_point, end_point, color, 2)
                cv2.circle(image, start_point, circle_radius, color, -1)
                cv2.circle(image, end_point, circle_radius, color, -1)

    def process_frame(self, image, pose):
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换BGR到RGB
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True

        # Update global variables for image dimensions
        self.image_width = image.shape[1]
        self.image_height = image.shape[0]

        keypoints = []
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            keypoints = [(lm.x, lm.y, lm.z) for lm in landmarks]

            self.draw_skeleton(image, keypoints, self.mp_pose.POSE_CONNECTIONS, (0, 255, 0))

            if self.recording:
                self.keypoints_data.append(keypoints)

        image = Image.fromarray(image)
        return image

    def analyze_video(self):
        self.new_frame = False
        self.frame_to_show = None

        self.cap = cv2.VideoCapture(self.video_path)
        self.keypoints_data = []
        self.video_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_playing = True
        self.start_time = time.time()

        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while self.cap.isOpened() and self.video_playing:
                if self.paused:
                    root.update()
                    continue

                times = [time.time()]

                if not self.dragging:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                ret, frame = self.cap.read()
                if not ret:
                    break

                times.append(time.time())  # Read frame time
                if not self.dragging:
                    self.current_frame += 1

                image = self.process_frame(frame, pose)
                times.append(time.time())  # Process frame time

                self.update_video_panel(image, video_panel)
                times.append(time.time())  # Update panel time

                self.update_progress_bar()
                times.append(time.time())  # Update progress bar time

                root.update()
                times.append(time.time())  # Update GUI time

                # 计算每一步的耗时
                durations = [times[i] - times[i - 1] for i in range(1, len(times))]
                steps = ["Read frame", "Process frame", "Update panel", "Update progress bar", "Update GUI"]

                for step, duration in zip(steps, durations):
                    print(f"{step} time: {duration:.4f}s")

        self.video_playing = False
        self.cap.release()
        cv2.destroyAllWindows()

    def save_annotations_to_csv(self):
        csv_file = os.path.splitext(self.video_path)[0] + '_label.csv'
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Start Frame', 'End Frame', 'Action Type', 'Action Name', 'Keypoints'])
            for annotation in self.annotations:
                writer.writerow(annotation)

    def load_annotations_from_csv(self):
        csv_file = os.path.splitext(self.video_path)[0] + '_label.csv'
        if os.path.exists(csv_file):
            with open(csv_file, mode='r') as file:
                reader = csv.reader(file)
                next(reader)
                self.annotations = [[int(row[0]), int(row[1]), row[2], row[3], row[4]] for row in reader]

    def update_video_panel(self, image, panel):
        panel_width = panel.winfo_width()
        panel_height = panel.winfo_height()

        if panel_width > 0 and panel_height > 0:
            image = self.resize_image_with_aspect_ratio(image, panel_width, panel_height)
            image = ImageTk.PhotoImage(image=image)
            panel.config(image=image)
            panel.image = image

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

    def update_progress_bar(self):
        if self.video_length > 0:
            progress = (self.current_frame / self.video_length) * 100
            progress_var.set(progress)

class PoseAnnotationApp:
    def __init__(self, root, pose_annotator):
        self.root = root
        self.pose_annotator = pose_annotator
        self.pose_annotator.app = self  # 将 PoseAnnotationApp 实例赋值给 PoseAnnotator 的 app 属性

        self.setup_ui()

    def setup_ui(self):
        self.root.title("Pose Annotation Tool")
        self.root.state('zoomed')
        global screen_width, screen_height, left_frame_width, right_frame_width, progress_var
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        left_frame_width = int(screen_width * 0.68)
        right_frame_width = screen_width - left_frame_width

        self.left_frame = ttk.Frame(self.root, width=left_frame_width, height=screen_height)
        self.left_frame.pack(side="left", fill="y", expand=False)

        instructions = (
            "F1: Open video file\n"
            "b: Start annotation period\n"
            "e: Stop annotation period\n"
            "d: Delete selected annotation\n"
            "Space: Pause/Resume video play\n"
            "Right Arrow: Fast forward 30 frames\n"
            "Left Arrow: Fast backward 30 frames\n"
            "Drag progress bar: Control video playback\n"
            "Double-click annotation: Jump to annotation segment\n"
            "Escape: Exit application"
        )

        # Create a frame to hold two columns of instructions
        title_frame = tk.Frame(self.left_frame, bg="black")
        title_frame.pack(side="top", fill="x")

        # Split the instruction text into two columns
        instructions_lines = instructions.split('\n')
        col1 = "\n".join(instructions_lines[:5])  # First 4 lines in the first column
        col2 = "\n".join(instructions_lines[5:])  # Remaining lines in the second column

        title_label_col1 = tk.Label(title_frame, text=col1, font=("Arial", 16), bg="black", fg="green", anchor="w",
                                    justify="left")
        title_label_col1.pack(side="left", fill="x", expand=True)

        title_label_col2 = tk.Label(title_frame, text=col2, font=("Arial", 16), bg="black", fg="green", anchor="w",
                                    justify="left")
        title_label_col2.pack(side="left", fill="x", expand=True)

        # 调整视频面板位置和大小
        video_aspect_ratio = 16 / 9  # 视频的宽高比，例如16:9
        video_panel_width = left_frame_width
        video_panel_height = int(video_panel_width / video_aspect_ratio)
        self.video_panel_frame = ttk.Frame(self.left_frame, width=video_panel_width, height=video_panel_height)
        self.video_panel_frame.pack(side="top", fill="both", expand=True)
        self.video_panel_frame.pack_propagate(0)

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

        self.annotation_listbox = tk.Listbox(self.right_frame)
        self.annotation_listbox.pack(side="top", fill="both", expand=True)
        self.annotation_listbox.bind('<<ListboxSelect>>', self.on_annotation_select)

        global video_panel
        video_panel = tk.Label(self.video_panel_frame)
        video_panel.pack(fill="both", expand=True)
        video_panel.pack_propagate(0)

        self.root.bind("<KeyPress>", self.on_key_press)

    def on_key_press(self, event):
        if event.keysym == 'Escape':
            self.pose_annotator.stop_video_playback()
            self.root.destroy()
            cv2.destroyAllWindows()
        elif event.keysym == 'F1':
            self.open_video_file()
        elif event.keysym == 'b':
            self.start_recording()
        elif event.keysym == 'e':
            self.stop_recording()
        elif event.keysym == 'd':
            self.delete_selected_annotation()
        elif event.keysym == 'Right':
            self.fast_forward()
        elif event.keysym == 'Left':
            self.rewind()
        elif event.keysym == 'space':
            self.toggle_pause()

    def toggle_pause(self):
        if self.pose_annotator.video_playing:
            if not self.pose_annotator.paused:
                self.pose_annotator.paused = True
            else:
                self.pose_annotator.paused = False
                self.pose_annotator.analyze_video()

    def fast_forward(self):
        self.pose_annotator.current_frame = min(self.pose_annotator.current_frame + 30, self.pose_annotator.video_length - 1)
        self.update_video_to_frame(self.pose_annotator.current_frame)

    def rewind(self):
        self.pose_annotator.current_frame = max(self.pose_annotator.current_frame - 30, 0)
        self.update_video_to_frame(self.pose_annotator.current_frame)

    def open_video_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if file_path:
            self.pose_annotator.stop_video_playback()  # 停止当前视频播放
            self.pose_annotator.reset_variables()  # 重置变量
            self.pose_annotator.video_path = file_path
            self.pose_annotator.load_annotations_from_csv()
            self.update_annotation_list()
            self.pose_annotator.analyze_video()

    def start_recording(self):
        self.pose_annotator.recording = True
        self.pose_annotator.start_frame = self.pose_annotator.current_frame
        self.pose_annotator.start_time = time.time()
        print("Recording started")

    def stop_recording(self):
        self.pose_annotator.recording = False
        self.pose_annotator.end_frame = self.pose_annotator.current_frame
        self.pose_annotator.paused = True
        self.pose_annotator.pause_frame = self.pose_annotator.current_frame
        end_time = time.time()
        duration = end_time - self.pose_annotator.start_time
        print("Recording stopped")
        if self.pose_annotator.keypoints_data:
            input_dialog = ActionInputDialog(self.root, self.pose_annotator.config)
            self.root.wait_window(input_dialog.dialog)
            if input_dialog.action_name and input_dialog.action_type:
                annotation = [
                    self.pose_annotator.start_frame,
                    self.pose_annotator.end_frame,
                    input_dialog.action_type,
                    input_dialog.action_name,
                    self.pose_annotator.keypoints_data
                ]
                self.pose_annotator.annotations.append(annotation)
                self.pose_annotator.keypoints_data = []
                self.pose_annotator.save_annotations_to_csv()
                self.update_annotation_list()
        # 继续播放视频
        self.pose_annotator.paused = False
        self.pose_annotator.current_frame = self.pose_annotator.pause_frame
        self.pose_annotator.analyze_video()

    def update_annotation_list(self):
        self.annotation_listbox.delete(0, tk.END)
        for annotation in self.pose_annotator.annotations:
            start_frame, end_frame, action_type, action_name, _ = annotation
            listbox_entry = f"Start: {start_frame}, End: {end_frame}, Type: {action_type}, Name: {action_name}, Frames: {end_frame - start_frame}"
            self.annotation_listbox.insert(tk.END, listbox_entry)

    def on_annotation_select(self, event):
        selection = event.widget.curselection()
        if selection:
            index = selection[0]
            annotation = self.pose_annotator.annotations[index]
            self.pose_annotator.current_frame = int(annotation[0])
            self.update_video_to_frame(self.pose_annotator.current_frame)

    def delete_selected_annotation(self):
        selection = self.annotation_listbox.curselection()
        if selection:
            index = selection[0]
            del self.pose_annotator.annotations[index]
            self.pose_annotator.save_annotations_to_csv()
            self.update_annotation_list()

    def update_video_to_frame(self, frame_number):
        self.pose_annotator.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.pose_annotator.cap.read()
        if ret:
            with self.pose_annotator.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                image = self.pose_annotator.process_frame(frame, pose)
                self.pose_annotator.update_video_panel(image, video_panel)

    def on_progress_bar_drag(self, value):
        self.pose_annotator.dragging = True

    def on_progress_bar_release(self, event):
        self.pose_annotator.dragging = False
        self.pose_annotator.current_frame = int(
            (progress_var.get() / 100) * self.pose_annotator.video_length)  # 更新当前帧位置
        self.update_video_to_frame(self.pose_annotator.current_frame)

    def on_progress_bar_press(self, event):
        self.pose_annotator.dragging = True

class ActionInputDialog:
    def __init__(self, parent, config):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Input Action Information")
        self.dialog.geometry("400x300")
        self.dialog.grab_set()
        self.config = config

        screen_width = self.dialog.winfo_screenwidth()
        screen_height = self.dialog.winfo_screenheight()
        x = (screen_width - 400) // 2
        y = (screen_height - 300) // 2
        self.dialog.geometry(f"+{x}+{y}")

        frame = ttk.Frame(self.dialog, padding="10")
        frame.pack(fill="both", expand=True)

        left_frame = ttk.Frame(frame)
        left_frame.pack(side="left", fill="y", expand=False)

        right_frame = ttk.Frame(frame)
        right_frame.pack(side="left", fill="both", expand=True)

        tk.Label(left_frame, text="Action Type:").pack(pady=5, anchor='w')
        self.action_type_var = tk.StringVar(value=self.config["default_action_type"])

        action_type_frame = ttk.Frame(right_frame)
        action_type_frame.pack(fill="x")
        for action_type in self.config["action_types"].keys():
            tk.Radiobutton(action_type_frame, text=action_type, variable=self.action_type_var, value=action_type, command=self.update_action_names).pack(anchor="w")

        tk.Label(left_frame, text="Action Name:").pack(pady=5, anchor='w')
        self.action_name_var = tk.StringVar()

        self.action_name_frame = ttk.Frame(right_frame)
        self.action_name_frame.pack(fill="x", pady=5)
        self.update_action_names()

        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(fill="x", pady=5)

        self.save_button = tk.Button(button_frame, text="Save", command=self.save)
        self.save_button.pack(side="left", padx=10)

        self.cancel_button = tk.Button(button_frame, text="Cancel", command=self.dialog.destroy)
        self.cancel_button.pack(side="right", padx=10)

        self.action_name = None
        self.action_type = None

    def update_action_names(self):
        for widget in self.action_name_frame.winfo_children():
            widget.destroy()
        action_type = self.action_type_var.get()
        actions = self.config["action_types"][action_type]["actions"]
        default_action = self.config["action_types"][action_type]["default_action"]
        self.action_name_var.set(default_action)
        for action_name in actions:
            tk.Radiobutton(self.action_name_frame, text=action_name, variable=self.action_name_var, value=action_name).pack(anchor="w")

    def save(self):
        self.action_name = self.action_name_var.get()
        self.action_type = self.action_type_var.get()
        self.dialog.destroy()




if __name__ == "__main__":
    root = tk.Tk()
    pose_annotator = PoseAnnotator()
    app = PoseAnnotationApp(root, pose_annotator)
    root.mainloop()
