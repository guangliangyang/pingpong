import cv2
from skimage import filters
import os

# 设置视频路径
video_path = os.path.join('..', 'mp4', '01.mov')

# 打开视频文件
cap = cv2.VideoCapture(video_path)

# 检查视频是否打开成功
if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

# 获取视频的帧率
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Frames per second: {fps}")

# 逐帧处理视频
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 将帧转换为灰度图像
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 使用 Scikit-Image 进行图像处理（例如边缘检测）
    edges = filters.sobel(gray_frame)

    # 将处理结果转换为 OpenCV 格式
    edges_uint8 = (edges * 255).astype('uint8')

    # 显示处理后的帧
    cv2.imshow('Processed Frame', edges_uint8)

    # 按 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频捕获对象并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
