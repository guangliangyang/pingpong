import cv2
import numpy as np

# 假设我们已经通过标定获得了摄像机的内参和外参
# 摄像机1的内参和外参
camera_matrix_1 = np.array([[fx1, 0, cx1],
                            [0, fy1, cy1],
                            [0,  0,  1]], dtype=np.float32)
dist_coeffs_1 = np.zeros((4, 1))  # 假设没有畸变，或已经校正
rotation_matrix_1 = np.eye(3)  # 假设摄像机1作为参考系，所以是单位矩阵
translation_vector_1 = np.zeros((3, 1))  # 摄像机1的平移向量为零

# 摄像机2的内参和外参
camera_matrix_2 = np.array([[fx2, 0, cx2],
                            [0, fy2, cy2],
                            [0,  0,  1]], dtype=np.float32)
dist_coeffs_2 = np.zeros((4, 1))  # 假设没有畸变，或已经校正
rotation_matrix_2 = np.array([[r11, r12, r13],
                              [r21, r22, r23],
                              [r31, r32, r33]], dtype=np.float32)
translation_vector_2 = np.array([[tx], [ty], [tz]], dtype=np.float32)

# 构建左右摄像机的投影矩阵
P1 = camera_matrix_1 @ np.hstack((rotation_matrix_1, translation_vector_1))
P2 = camera_matrix_2 @ np.hstack((rotation_matrix_2, translation_vector_2))

# 读取左右摄像机图像
img1 = cv2.imread('left_image.png')
img2 = cv2.imread('right_image.png')

# 转换为灰度图像
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 使用Hough变换检测圆形（假设乒乓球是一个圆形）
circles1 = cv2.HoughCircles(gray1, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                            param1=100, param2=30, minRadius=10, maxRadius=50)
circles2 = cv2.HoughCircles(gray2, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                            param1=100, param2=30, minRadius=10, maxRadius=50)

if circles1 is not None and circles2 is not None:
    # 假设找到了乒乓球的圆心，提取圆心坐标
    x1, y1, r1 = circles1[0][0]  # 左图像中的乒乓球圆心坐标
    x2, y2, r2 = circles2[0][0]  # 右图像中的乒乓球圆心坐标

    # 将乒乓球的2D坐标转换为齐次坐标形式
    point_2d_cam1 = np.array([[x1], [y1]], dtype=np.float32)
    point_2d_cam2 = np.array([[x2], [y2]], dtype=np.float32)

    # 使用三角测量法恢复乒乓球的3D点
    point_4d_homogeneous = cv2.triangulatePoints(P1, P2, point_2d_cam1, point_2d_cam2)

    # 将齐次坐标转换为3D点
    point_3d = point_4d_homogeneous[:3] / point_4d_homogeneous[3]

    print("Reconstructed 3D point of the ping-pong ball:\n", point_3d)
else:
    print("未能检测到乒乓球的圆形特征。")
