import cv2
import numpy as np
from ultralytics import YOLO

# 加载YOLO模型
model_path = "/Users/andy/workspace/AUT-PY/Table_tennis_demo_zh/best_bak.pt"
model = YOLO(model_path)

# 读取视频
video_path = "/Users/andy/workspace/GX010028-spin.MP4"
cap = cv2.VideoCapture(video_path)

# 初始化一个空白图像来合成乒乓球图像
combined_image = None
frame_count = 0
desired_size = (50, 50)  # 设置所有图像调整到的大小

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 使用YOLO模型进行检测
    results = model(frame)

    # 解析检测结果，过滤掉类别标签为 3 的检测框
    for result in results[0].boxes:
        if int(result.cls) == 3:  # 类别标签为 3 的检测框
            x1, y1, x2, y2 = map(int, result.xyxy[0])

            # 裁剪出乒乓球区域
            pingpong_ball = frame[y1:y2, x1:x2]

            # 调整乒乓球图像到目标大小
            if pingpong_ball.size > 0:
                pingpong_ball = cv2.resize(pingpong_ball, desired_size)
                if combined_image is None:
                    combined_image = pingpong_ball
                else:
                    combined_image = np.hstack((combined_image, pingpong_ball))

    frame_count += 1

cap.release()

# 保存最终合成的图像
output_path = "/Users/andy/workspace/combined_pingpong_image.png"
cv2.imwrite(output_path, combined_image)
