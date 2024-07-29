import tkinter as tk
import cv2
import pygame
import numpy as np
import os
import time
import mediapipe as mp


class VideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        self.running = False

        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils

        # Get video frame rate
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.delay = int(1000 / self.fps)  # Delay in milliseconds between frames

    def start(self):
        self.running = True
        self.start_time = time.time()
        self.frame_count = 0
        self.process_video()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def process_video(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = self.process_frame(frame)
            app.update_pygame_display(frame)

            # Calculate the time to wait until the next frame should be displayed
            self.frame_count += 1
            elapsed_time = (time.time() - self.start_time) * 1000  # Convert to milliseconds
            expected_time = self.frame_count * self.delay
            wait_time = max(1, int(expected_time - elapsed_time))

            print("waited {} ms".format(wait_time))
            root.after(wait_time, self.process_video)  # Schedule the next frame
        else:
            self.stop()  # Stop if the video is done

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        # Draw pose landmarks
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame_rgb, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        return frame_rgb


class PoseEstimationApp:
    def __init__(self, root, video_path):
        self.root = root
        self.root.title("Pose Estimation with Pygame")

        # Set the window size to a small rectangular size
        self.root.geometry("300x100")

        self.video_processor = VideoProcessor(video_path)

        # Initialize pygame
        pygame.init()
        self.screen_width = self.root.winfo_screenwidth() // 2
        self.screen_height = self.root.winfo_screenheight() // 2
        self.pygame_position = 'bottom_left'
        self.set_pygame_position()

        # Add buttons to control the video display
        self.button_frame = tk.Frame(root)
        self.button_frame.pack(side="bottom", anchor="sw")

        self.start_button = tk.Button(self.button_frame, text="Start Video", command=self.start_video)
        self.start_button.pack(side="left")

        self.stop_button = tk.Button(self.button_frame, text="Stop Video", command=self.stop_video)
        self.stop_button.pack(side="left")

        self.quit_button = tk.Button(self.button_frame, text="Quit", command=self.quit_app)
        self.quit_button.pack(side="right")

        # Bind F4 key to toggle pygame window position
        self.root.bind("<F4>", self.toggle_pygame_position)

    def start_video(self):
        if not self.video_processor.running:
            self.video_processor.start()

    def stop_video(self):
        if self.video_processor.running:
            self.video_processor.stop()
            pygame.quit()

    def quit_app(self):
        self.stop_video()
        self.root.destroy()

    def update_pygame_display(self, frame):
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally

        # Maintain aspect ratio while resizing the frame
        frame_height, frame_width = frame.shape[:2]
        window_width, window_height = self.screen.get_size()

        # Calculate the scaling factor to maintain aspect ratio
        scale = min(window_width / frame_width, window_height / frame_height)
        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)

        # Resize the frame to fit the window while maintaining aspect ratio
        frame_resized = cv2.resize(frame, (new_width, new_height))

        # Create a surface with the same size as the Pygame window
        frame_surface = pygame.Surface((window_width, window_height))

        # Fill the surface with black color
        frame_surface.fill((0, 0, 0))

        # Convert the resized frame to a Pygame surface
        frame_resized = np.rot90(frame_resized)
        frame_resized = pygame.surfarray.make_surface(frame_resized)

        # Blit the resized frame onto the center of the Pygame window surface
        frame_surface.blit(frame_resized, ((window_width - new_width) // 2, (window_height - new_height) // 2))

        # Update the Pygame display with the new surface
        self.screen.blit(frame_surface, (0, 0))
        pygame.display.update()

    def toggle_pygame_position(self, event):
        if self.pygame_position == 'bottom_left':
            self.pygame_position = 'top_right'
        else:
            self.pygame_position = 'bottom_left'
        self.set_pygame_position()

    def set_pygame_position(self):
        if self.pygame_position == 'bottom_left':
            x = 0
            y = self.root.winfo_screenheight() - self.screen_height
        else:
            x = self.root.winfo_screenwidth() - self.screen_width
            y = 0
        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x},{y}"
        pygame.display.quit()
        pygame.display.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Video Display")


if __name__ == "__main__":
    video_path = os.path.join('..', 'mp4', '01.mov')
    root = tk.Tk()
    app = PoseEstimationApp(root, video_path)
    root.mainloop()
