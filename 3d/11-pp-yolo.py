import cv2
import numpy as np

# 已知的乒乓球台关键点的3D坐标（单位：米，具体值根据实际场景填写）
object_points = np.array([
    [x1, y1, z1],  # 台面第1个角点
    [x2, y2, z2],  # 台面第2个角点
    [x3, y3, z3],  # 台面第3个角点
    [x4, y4, z4],  # 台面第4个角点
    [x5, y5, z5],  # 球网第1个点
    [x6, y6, z6],  # 球网第2个点
], dtype=np.float32)

# 使用YOLO检测到的关键点的2D像素坐标
image_points_cam1 = np.array([
    [u1_1, v1_1],  # 第1个角点在摄像机1中的像素坐标
    [u2_1, v2_1],
    # ... 其他关键点
], dtype=np.float32)

image_points_cam2 = np.array([
    [u1_2, v1_2],  # 第1个角点在摄像机2中的像素坐标
    [u2_2, v2_2],
    # ... 其他关键点
], dtype=np.float32)

# 假设我们已经初始化了摄像机的内参矩阵
camera_matrix_cam1 = np.array([[fx1, 0, cx1],
                               [0, fy1, cy1],
                               [0,  0,  1]], dtype=np.float32)

camera_matrix_cam2 = np.array([[fx2, 0, cx2],
                               [0, fy2, cy2],
                               [0,  0,  1]], dtype=np.float32)

dist_coeffs_cam1 = np.zeros((4, 1))  # 假设无畸变，或已经校正
dist_coeffs_cam2 = np.zeros((4, 1))

# 通过PnP算法计算每个摄像机的外参
ret_cam1, rvec_cam1, tvec_cam1 = cv2.solvePnP(object_points, image_points_cam1, camera_matrix_cam1, dist_coeffs_cam1)
ret_cam2, rvec_cam2, tvec_cam2 = cv2.solvePnP(object_points, image_points_cam2, camera_matrix_cam2, dist_coeffs_cam2)

# 将旋转向量转换为旋转矩阵
rotation_matrix_cam1, _ = cv2.Rodrigues(rvec_cam1)
rotation_matrix_cam2, _ = cv2.Rodrigues(rvec_cam2)

# 构建摄像机的投影矩阵
P1 = camera_matrix_cam1 @ np.hstack((rotation_matrix_cam1, tvec_cam1))
P2 = camera_matrix_cam2 @ np.hstack((rotation_matrix_cam2, tvec_cam2))

# 使用YOLO检测乒乓球的2D像素坐标
ball_2d_cam1 = np.array([[u_ball_cam1, v_ball_cam1]], dtype=np.float32).T
ball_2d_cam2 = np.array([[u_ball_cam2, v_ball_cam2]], dtype=np.float32).T

# 使用三角测量法恢复乒乓球的3D坐标
ball_4d_homogeneous = cv2.triangulatePoints(P1, P2, ball_2d_cam1, ball_2d_cam2)
ball_3d = ball_4d_homogeneous[:3] / ball_4d_homogeneous[3]

print("Reconstructed 3D position of the ping-pong ball:", ball_3d)


'''
具体步骤：
使用YOLO检测乒乓球台关键点：

利用YOLO模型检测乒乓球台的关键点，例如台面四个角点和球网的两端点。
对每个摄像机，获取这些关键点的2D像素坐标。
使用PnP算法计算每个摄像机的内外参：

已知乒乓球台关键点的3D坐标（例如通过物理测量得到）。
使用YOLO检测到的2D像素坐标，通过OpenCV的solvePnP函数计算每个摄像机的内外参。
这些内外参将用于3D空间建模。
将两个摄像机的坐标系对齐到同一个世界坐标系：

选择一个摄像机的坐标系作为世界坐标系，或者将乒乓球台中心定义为世界坐标系的原点。
使用两个摄像机之间的相对位姿关系来转换它们的坐标系。
利用YOLO检测到的乒乓球位置计算3D坐标：

在两个摄像机图像中使用YOLO检测乒乓球的2D像素坐标。
使用两个摄像机的投影矩阵和YOLO检测到的2D点，通过三角测量法计算乒乓球的3D坐标。

'''