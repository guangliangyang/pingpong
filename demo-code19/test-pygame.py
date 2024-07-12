import os
import cv2
import pygame
import numpy as np
import mediapipe as mp
from PIL import Image
from ultralytics import YOLOv10


class VideoProcessor:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("Error: Cannot open video.")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        self.cap.release()


class PygameApp:
    def __init__(self, video_path):
        pygame.init()
        self.video_processor = VideoProcessor(video_path)
        self.screen = pygame.display.set_mode((self.video_processor.width, self.video_processor.height))
        pygame.display.set_caption("Video Processing with Pygame")
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False
        self.mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

    def process_frame(self, frame):
        # 转换为 RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 处理帧
        results = self.mp_pose.process(image_rgb)

        # 绘制骨架
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

        # 在图像上绘制自定义线框
        self.draw_custom_lines(frame)

        # 将处理后的图像转换为 Pygame 表面
        frame_surface = pygame.surfarray.make_surface(frame.transpose((1, 0, 2)))
        return frame_surface

    def draw_custom_lines(self, image):
        # 自定义线框的顶点坐标
        points = np.array([[100, 100], [200, 100], [200, 200], [100, 200]])
        points2 = np.array([[300, 300], [400, 300], [400, 400], [300, 400]])

        # 绘制线框
        cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.polylines(image, [points2], isClosed=True, color=(255, 0, 0), thickness=2)

    def handle_keys(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:
            self.running = False
        if keys[pygame.K_p]:
            self.paused = not self.paused

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            self.handle_keys()

            if not self.paused:
                frame = self.video_processor.get_frame()
                if frame is None:
                    break

                processed_frame = self.process_frame(frame)
                self.screen.blit(processed_frame, (0, 0))

            pygame.display.flip()
            self.clock.tick(self.video_processor.fps)

        self.video_processor.release()
        pygame.quit()


if __name__ == "__main__":
    video_path = os.path.join('..', 'mp4', '01.mov')
    app = PygameApp(video_path)
    app.run()
