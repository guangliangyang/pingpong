import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 定义棋盘格的尺寸
chessboard_size = (8, 8)  # 内角点数量
square_size = 10.0  # 每个格子的实际大小为10cm

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
ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

if ret:
    # 精细化角点位置
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # 可视化角点
    cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title('Detected Chessboard Corners')
    plt.show()
else:
    raise ValueError("无法检测到棋盘格角点")

# 定义棋盘在物理空间中的实际坐标
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# 计算相机标定参数
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp], [corners], gray.shape[::-1], None, None)

# 打印相机标定结果
print("Camera Matrix:\n", mtx)
print("Distortion Coefficients:\n", dist)

# 定义目标点在图像中的位置
target_dst = np.float32([
    [500, 285],  # 左上角
    [850, 300],  # 右上角
    [1225, 500], # 右下角
    [415, 502]   # 左下角
])

# 使用相机标定参数进行透视变换
def undistort_points(points, mtx, dist):
    points = np.expand_dims(points, axis=1)
    undistorted_points = cv2.undistortPoints(points, mtx, dist, P=mtx)
    return undistorted_points.reshape(-1, 2)

undistorted_target_dst = undistort_points(target_dst, mtx, dist)

# 打印未畸变的图像坐标
print("Undistorted Image Points:\n", undistorted_target_dst)

# 使用PnP算法计算3D坐标
_, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist)
rotation_matrix, _ = cv2.Rodrigues(rvec)

# 将目标点转换为3D坐标
def image_to_world(image_points, rotation_matrix, tvec, mtx):
    image_points_hom = cv2.convertPointsToHomogeneous(image_points)[:, 0, :]
    world_points_hom = np.dot(np.linalg.inv(rotation_matrix), (np.dot(np.linalg.inv(mtx), image_points_hom.T) - tvec)).T
    return world_points_hom

world_points = image_to_world(undistorted_target_dst, rotation_matrix, tvec, mtx)

# 打印转换后的3D坐标
print("Transformed World Points (in cm):\n", world_points)

# 可视化结果
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.scatter(target_dst[:, 0], target_dst[:, 1], color='red', marker='o')
for i, txt in enumerate(world_points):
    plt.annotate(f'({txt[0]:.2f}, {txt[1]:.2f}, {txt[2]:.2f})', (target_dst[i][0], target_dst[i][1]), color='yellow')
plt.title('Image with Target Points and 3D Coordinates')
plt.show()
