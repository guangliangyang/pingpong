import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageOps

# 初始化Mediapipe的姿态检测和绘图工具
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# 绘制关键点及其连接并显示索引
def draw_landmarks_with_connections(image, keypoints, connections, color):
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        if start_idx < len(keypoints) and end_idx < len(keypoints) and start_idx > 10 and end_idx > 10:
            start_point = (int(keypoints[start_idx][0] * image.shape[1]), int(keypoints[start_idx][1] * image.shape[0]))
            end_point = (int(keypoints[end_idx][0] * image.shape[1]), int(keypoints[end_idx][1] * image.shape[0]))
            cv2.line(image, start_point, end_point, color, 2)
            cv2.circle(image, start_point, 5, color, -1)
            cv2.circle(image, end_point, 5, color, -1)
            cv2.putText(image, str(start_idx), start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
            cv2.putText(image, str(end_idx), end_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


def draw_face_triangle(image, keypoints):
    # 面部关键点索引
    left_eye_outer_index = 3
    right_eye_outer_index = 6
    mouth_left_index = 10
    mouth_right_index = 9

    if all(idx < len(keypoints) for idx in
           [left_eye_outer_index, right_eye_outer_index, mouth_left_index, mouth_right_index]):
        # 获取面部关键点的坐标
        left_eye_outer_point = keypoints[left_eye_outer_index][:2]
        right_eye_outer_point = keypoints[right_eye_outer_index][:2]
        mouth_left_point = keypoints[mouth_left_index][:2]
        mouth_right_point = keypoints[mouth_right_index][:2]

        # 将坐标转换为图像上的像素坐标
        left_eye_outer_point = (
        int(left_eye_outer_point[0] * image.shape[1]), int(left_eye_outer_point[1] * image.shape[0]))
        right_eye_outer_point = (
        int(right_eye_outer_point[0] * image.shape[1]), int(right_eye_outer_point[1] * image.shape[0]))
        mouth_left_point = (int(mouth_left_point[0] * image.shape[1]), int(mouth_left_point[1] * image.shape[0]))
        mouth_right_point = (int(mouth_right_point[0] * image.shape[1]), int(mouth_right_point[1] * image.shape[0]))

        # 绘制倒三角形
        triangle_cnt = np.array([left_eye_outer_point, right_eye_outer_point, mouth_left_point, mouth_right_point],
                                np.int32)
        triangle_cnt = triangle_cnt.reshape((-1, 1, 2))
        cv2.drawContours(image, [triangle_cnt], 0, (255, 255, 255), 2)  # 使用绿色绘制

        # 显示关键点索引
        cv2.putText(image, str(left_eye_outer_index), left_eye_outer_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                    2, cv2.LINE_AA)
        cv2.putText(image, str(right_eye_outer_index), right_eye_outer_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, str(mouth_left_index), mouth_left_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(image, str(mouth_right_index), mouth_right_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                    cv2.LINE_AA)


def draw_connection_line(image, keypoints):
    # 面部和身体关键点索引
    mouth_left_index = 9
    mouth_right_index = 10
    left_shoulder_index = 11
    right_shoulder_index = 12

    if all(idx < len(keypoints) for idx in
           [mouth_left_index, mouth_right_index, left_shoulder_index, right_shoulder_index]):
        # 获取面部和身体关键点的坐标
        mouth_left_point = keypoints[mouth_left_index][:2]
        mouth_right_point = keypoints[mouth_right_index][:2]
        left_shoulder_point = keypoints[left_shoulder_index][:2]
        right_shoulder_point = keypoints[right_shoulder_index][:2]

        # 计算中间点
        mouth_mid_point = (
        (mouth_left_point[0] + mouth_right_point[0]) / 2, (mouth_left_point[1] + mouth_right_point[1]) / 2)
        shoulder_mid_point = (
        (left_shoulder_point[0] + right_shoulder_point[0]) / 2, (left_shoulder_point[1] + right_shoulder_point[1]) / 2)

        # 将坐标转换为图像上的像素坐标
        mouth_mid_point = (int(mouth_mid_point[0] * image.shape[1]), int(mouth_mid_point[1] * image.shape[0]))
        shoulder_mid_point = (int(shoulder_mid_point[0] * image.shape[1]), int(shoulder_mid_point[1] * image.shape[0]))

        # 绘制直线
        cv2.line(image, mouth_mid_point, shoulder_mid_point, (255, 255, 255), 2)

        # 显示关键点索引
        cv2.putText(image, "M", mouth_mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "S", shoulder_mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)


def update_data_panel(panel, keypoints):
    data_text = ""
    if keypoints:
        for idx, (x, y, z) in enumerate(keypoints):
            data_text += f"Point {idx}: x={x:.2f}, y={y:.2f}, z={z:.2f}\n"
    panel.config(text=data_text)


def resize_image_with_aspect_ratio(image, target_width, target_height):
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


# 设置摄像头
cap = cv2.VideoCapture(0)

# 创建Tkinter窗口
root = tk.Tk()
root.title("Pose Estimation and Analysis")

# 设置窗口最大化
root.state('zoomed')

# 获取屏幕尺寸
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# 计算左侧和右侧面板的宽度
left_frame_width = int(screen_width * 0.68)
right_frame_width = screen_width - left_frame_width

# 创建左侧监控画面
left_frame = ttk.Frame(root, width=left_frame_width, height=screen_height)
left_frame.pack(side="left", fill="y", expand=False)

video_panel = tk.Label(left_frame)
video_panel.pack(fill="both", expand=True)

# 创建右侧检测数据面板
right_frame = ttk.Frame(root, width=right_frame_width, height=screen_height)
right_frame.pack(side="right", fill="both", expand=True)

data_panel = tk.Label(right_frame, text="Detection Data", justify=tk.LEFT, anchor="nw")
data_panel.pack(side="top", fill="both", expand=True)


def update_video_panel(image):
    # 获取窗口尺寸
    window_width = left_frame_width
    window_height = screen_height

    # 调整图像大小以适应窗口并保持比例
    if window_width > 0 and window_height > 0:
        image = resize_image_with_aspect_ratio(image, window_width, window_height)
        image = ImageTk.PhotoImage(image=image)

        video_panel.config(image=image)
        video_panel.image = image


# 使用Mediapipe的Pose模块处理视频流
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    def update_frame():
        ret, frame = cap.read()
        if ret:
            # 转换颜色空间
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # 处理图像，获取姿态关键点
            results = pose.process(image)

            # 转回BGR颜色空间
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            keypoints = []

            # 绘制姿态关键点
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # 获取关键点坐标
                keypoints = [(lm.x, lm.y, lm.z) for lm in landmarks]

                # 绘制身体关键点和连接线并显示索引
                draw_landmarks_with_connections(image, keypoints, mp_pose.POSE_CONNECTIONS, (255, 255, 255))

                # 绘制面部倒三角形并显示索引
                draw_face_triangle(image, keypoints)

                # 绘制面部到身体的连接线
                draw_connection_line(image, keypoints)

            # 更新数据面板
            update_data_panel(data_panel, keypoints)

            # 将图像转换为PIL格式
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

            # 更新视频面板
            update_video_panel(image)

        root.after(10, update_frame)


    # 开始更新帧
    root.after(10, update_frame)


    def on_key_press(event):
        if event.keysym == 'Escape':
            root.destroy()
            cap.release()
            cv2.destroyAllWindows()


    root.bind("<KeyPress>", on_key_press)
    root.mainloop()
