import tkinter as tk
import threading
import time
from queue import Queue


class PoseApp:
    def __init__(self, root, pose_estimation):
        self.root = root
        self.pose_estimation = pose_estimation
        self.data_queue = Queue()
        self.speed_queue = Queue()
        self.setup_ui()

        # 启动独立线程来处理繁重任务
        threading.Thread(target=self.data_updater, daemon=True).start()
        threading.Thread(target=self.speed_updater, daemon=True).start()

        # 启动 UI 更新循环
        self.root.after(1000, self.update_ui)

    def setup_ui(self):
        self.label = tk.Label(self.root, text="Starting...")
        self.label.pack()
        self.speed_label = tk.Label(self.root, text="Speed...")
        self.speed_label.pack()

    def update_ui(self):
        start_time = time.time()

        # 批量更新界面
        self.update_data_from_queue()
        self.update_speed_from_queue()

        # 调整更新频率
        elapsed_time = time.time() - start_time
        update_interval = max(1000 - int(elapsed_time * 1000), 50)
        self.root.after(update_interval, self.update_ui)

    def update_data_from_queue(self):
        try:
            while not self.data_queue.empty():
                data = self.data_queue.get_nowait()
                self.label.config(text=data)
        except Queue.Empty:
            pass

    def update_speed_from_queue(self):
        try:
            while not self.speed_queue.empty():
                speed = self.speed_queue.get_nowait()
                self.speed_label.config(text=speed)
        except Queue.Empty:
            pass

    def data_updater(self):
        while True:
            # 模拟繁重任务
            time.sleep(5)
            data = "Updated Data: " + time.ctime()
            self.data_queue.put(data)

    def speed_updater(self):
        while True:
            # 模拟速度更新任务
            time.sleep(1)
            speed = "Updated Speed: " + time.ctime()
            self.speed_queue.put(speed)


if __name__ == "__main__":
    root = tk.Tk()
    pose_estimation = None  # 初始化你的 pose_estimation 对象
    app = PoseApp(root, pose_estimation)
    root.mainloop()
