import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 定义棋盘格的尺寸
small_chessboard_size = (8, 8)  # 每个小棋盘格的内角点数量
small_square_size = 10.0  # 每个小格子的实际大小为10cm
large_square_width = 100.0  # 每个大格子的实际宽度为100cm
large_square_height = 75.0  # 每个大格子的实际高度为75cm
vertical_offset = -40.0  # 原点向下移动30cm

# 从视频中提取帧
video_path = os.path.join('..', 'mp4', '01.mov')  # 替换为实际的视频文件名
cap = cv2.VideoCapture(video_path)

# 读取第一帧用于检测棋盘
ret, frame = cap.read()
cap.release()

if not ret:
    raise ValueError("无法读取视频帧")

# 转换为灰度图像
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# 找到棋盘格角点
ret, corners = cv2.findChessboardCorners(gray, small_chessboard_size, None)

if ret:
    # 精细化角点位置
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
else:
    raise ValueError("无法检测到棋盘格角点")

# 定义小棋盘在物理空间中的实际坐标
objp = np.zeros((small_chessboard_size[0] * small_chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:small_chessboard_size[0], 0:small_chessboard_size[1]].T.reshape(-1, 2)
objp *= small_square_size

# 计算相机标定参数
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp], [corners], gray.shape[::-1], None, None)

# 打印相机标定结果
print("Camera Matrix:\n", mtx)
print("Distortion Coefficients:\n", dist)

# 获取棋盘左下角的点作为原点
left_bottom_corner = corners[-1][0]  # 选择左下角的角点作为原点

# 定义大棋盘格在图像中的起始位置
num_large_squares_x = 3  # 大棋盘格在X方向的数量
num_large_squares_y = 6  # 大棋盘格在Y方向的数量

# 创建大棋盘格的物理坐标，以左下角为原点，并向下移动30cm
chessboard_physical_points = []
for i in range(num_large_squares_y + 1):
    for j in range(num_large_squares_x + 1):
        chessboard_physical_points.append([j * large_square_width, -i * large_square_height - vertical_offset, 0])

chessboard_physical_points = np.array(chessboard_physical_points, dtype=np.float32)

# 将物理坐标转换为图像坐标
def project_points(physical_points, rvec, tvec, mtx, dist):
    image_points, _ = cv2.projectPoints(physical_points, rvec, tvec, mtx, dist)
    return image_points.reshape(-1, 2)

chessboard_image_points_px = project_points(chessboard_physical_points, rvecs[0], tvecs[0], mtx, dist)

# 计算每个大格子的顶点在图像中的位置
chessboard_vertices = []
for i in range(num_large_squares_y):
    for j in range(num_large_squares_x):
        top_left = chessboard_image_points_px[i * (num_large_squares_x + 1) + j]
        top_right = chessboard_image_points_px[i * (num_large_squares_x + 1) + j + 1]
        bottom_right = chessboard_image_points_px[(i + 1) * (num_large_squares_x + 1) + j + 1]
        bottom_left = chessboard_image_points_px[(i + 1) * (num_large_squares_x + 1) + j]
        chessboard_vertices.append([top_left, top_right, bottom_right, bottom_left])

# 绘制大棋盘格
output_image = frame.copy()
for vertices in chessboard_vertices:
    pts = np.array(vertices, dtype=np.int32)
    cv2.polylines(output_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

# 显示结果
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.title('Large Chessboard Pattern')
plt.show()
