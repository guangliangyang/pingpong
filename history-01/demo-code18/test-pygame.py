import os
import sys
import pygame
import cv2
import mediapipe as mp
import numpy as np
import time

# 初始化pygame
pygame.init()

# 创建pygame窗口
screen = pygame.display.set_mode((1600, int(600 * 4 / 3)))
pygame.display.set_caption("Statistics of Footwork & Arm Swings of Table Tennis")

class PoseEstimation:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.cap = None

    def initialize_video_capture(self, source):
        self.cap = cv2.VideoCapture(source)

    def close_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def process_video(self, frame, pose):
        # 假设frame是一个numpy数组，形状为(height, width, 3)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = frame
        return image

class PoseApp:
    def __init__(self, pose_estimation):
        self.pose_estimation = pose_estimation
        self.pose_estimation.app = self
        self.mode = "video"

    def main_loop(self):
        clock = pygame.time.Clock()
        self.pose_estimation.initialize_video_capture(0)  # 使用摄像头
        self.pose_estimation.template_match_counts = {"Arm": {}, "Footwork": {}}

        with self.pose_estimation.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.pose_estimation.close_camera()
                        pygame.quit()
                        sys.exit()

                if self.pose_estimation.cap and self.pose_estimation.cap.isOpened():
                    ret, frame = self.pose_estimation.cap.read()
                    if ret:
                        image = self.pose_estimation.process_video(frame, pose)
                        frame_surface = pygame.surfarray.make_surface(image)
                        screen.blit(frame_surface, (0, 0))
                        pygame.display.update()

                clock.tick(30)  # 控制帧率

if __name__ == "__main__":
    pose_estimation = PoseEstimation()
    app = PoseApp(pose_estimation)
    app.main_loop()
