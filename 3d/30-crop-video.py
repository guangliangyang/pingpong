import cv2
import numpy as np

# 读取视频
video_path = 'speed-01.mov'
cap = cv2.VideoCapture(video_path)

# 读取第一帧
ret, frame = cap.read()
cap.release()

if not ret:
    print("无法读取视频文件")
else:
    # 将图像转换为灰度图像
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 创建一个二值化图像，将所有黑色像素（值为0）和非黑色像素分离
    _, binary_frame = cv2.threshold(gray_frame, 1, 255, cv2.THRESH_BINARY)

    # 垂直和水平方向上投影，检测黑色区域
    vertical_projection = np.sum(binary_frame, axis=1)
    horizontal_projection = np.sum(binary_frame, axis=0)

    # 查找垂直方向上内容区域的起始和结束行（非黑色区域）
    y_min = np.where(vertical_projection > 0)[0][0]
    y_max = np.where(vertical_projection > 0)[0][-1]

    # 查找水平方向上内容区域的起始和结束列（非黑色区域）
    x_min = np.where(horizontal_projection > 0)[0][0]
    x_max = np.where(horizontal_projection > 0)[0][-1]

    # 裁剪掉上下左右的黑色边框，保留内容区域
    cropped_frame = frame[y_min:y_max+1, x_min:x_max+1]

    # 显示裁剪后的图像
    cv2.imshow("Cropped Frame", cropped_frame)
    cv2.imwrite("cropped_frame.png", cropped_frame)

    # 切分左右两部分
    height, width, _ = cropped_frame.shape
    mid_point = width // 2

    left_image = cropped_frame[:, :mid_point]
    right_image = cropped_frame[:, mid_point:]

    # 输出左右图像的分辨率
    left_resolution = left_image.shape[:2]  # (height, width)
    right_resolution = right_image.shape[:2]  # (height, width)

    print(f"Left Image Resolution: {left_resolution}")
    print(f"Right Image Resolution: {right_resolution}")

    # 显示切分后的两张图片
    cv2.imshow("Left Image", left_image)
    cv2.imshow("Right Image", right_image)

    # 保存图片（可选）
    cv2.imwrite("left_image.png", left_image)
    cv2.imwrite("right_image.png", right_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
