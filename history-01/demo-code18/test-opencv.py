import tkinter as tk
from tkinter import Label
import cv2
from PIL import Image, ImageTk
import os
import time
import threading

# 导入os模块并拼接视频路径
video_path = os.path.join('..', 'mp4', '01.mov')

# 创建一个Tkinter窗口
root = tk.Tk()
root.title("Video Player")

# 创建一个Label用来显示视频帧
video_label = Label(root)
video_label.pack()

# 打开视频文件
cap = cv2.VideoCapture(0)

# 全局变量
buffer_1 = None
buffer_2 = None
active_buffer = 1
stop_event = threading.Event()
buffer_lock = threading.Lock()

def read_frame():
    global buffer_1, buffer_2, active_buffer
    while not stop_event.is_set():
        ret, new_frame = cap.read()
        if ret:
            with buffer_lock:
                if active_buffer == 1:
                    buffer_1 = new_frame
                    active_buffer = 2
                else:
                    buffer_2 = new_frame
                    active_buffer = 1
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        time.sleep(0.005)  # 更短的延迟以提高读取帧的频率

def update_frame():
    global buffer_1, buffer_2, active_buffer
    with buffer_lock:
        if active_buffer == 1:
            current_frame = buffer_2
        else:
            current_frame = buffer_1

    if current_frame is not None:
        start_time = time.time()
        # 将BGR转换为RGB
        rgb_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        convert_color_time = time.time() - start_time

        start_time = time.time()
        # 将OpenCV图像转换为PIL图像
        image = Image.fromarray(rgb_frame)
        fromarray_time = time.time() - start_time

        start_time = time.time()
        # 将PIL图像转换为ImageTk格式
        imgtk = ImageTk.PhotoImage(image=image)
        photoimage_time = time.time() - start_time

        start_time = time.time()
        # 在Label中显示图像
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        update_label_time = time.time() - start_time

        # 打印每一步的执行时间
        print(f"convert_color_time: {convert_color_time:.6f} seconds, "
              f"fromarray_time: {fromarray_time:.6f} seconds, "
              f"photoimage_time: {photoimage_time:.6f} seconds, "
              f"update_label_time: {update_label_time:.6f} seconds")

    # 每30毫秒更新一次帧
    root.after(30, update_frame)

# 启动读取帧的线程
thread = threading.Thread(target=read_frame)
thread.start()

# 启动帧更新
update_frame()

# 运行Tkinter主循环
root.mainloop()

# 停止读取帧的线程
stop_event.set()
thread.join()

# 释放视频捕获对象
cap.release()
