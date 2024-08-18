import json
import numpy as np
import cv2


# 从JSON文件中读取标定结果
def load_calibration_results(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    camera_matrix = np.array(data["camera_matrix"])
    dist_coeffs = np.array(data["dist_coeffs"]).flatten()
    rvec = np.array(data["rvecs"][0]).flatten()  # 旋转向量
    tvec = np.array(data["tvecs"][0]).flatten()  # 平移向量

    return camera_matrix, dist_coeffs, rvec, tvec


# 将旋转向量和平移向量转换为4x4齐次变换矩阵
def get_homogeneous_transform_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)  # 将旋转向量转换为旋转矩阵
    Rt = np.hstack((R, tvec.reshape(-1, 1)))  # 构建 [R|t] 矩阵
    homogeneous_matrix = np.vstack((Rt, [0, 0, 0, 1]))  # 转换为4x4齐次矩阵
    return homogeneous_matrix


# 将图像的四个角点转换为世界坐标
def get_image_corners_in_world(camera_matrix, dist_coeffs, rvec, tvec, image_size):
    h, w = image_size
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)  # 图像的四个角点
    corners_undistorted = cv2.undistortPoints(np.expand_dims(corners, axis=1), camera_matrix, dist_coeffs)  # 去畸变
    corners_undistorted = np.squeeze(corners_undistorted)

    # 将去畸变后的图像点转换为齐次坐标 [x, y, 0, 1]
    corners_homogeneous = np.hstack([corners_undistorted, np.zeros((4, 1)), np.ones((4, 1))])

    # 计算变换矩阵并求逆，将图像坐标投影回世界坐标
    transform_matrix = get_homogeneous_transform_matrix(rvec, tvec)
    transform_matrix_inv = np.linalg.inv(transform_matrix)  # 求逆矩阵

    # 将图像点转换为世界坐标
    world_corners = (transform_matrix_inv @ corners_homogeneous.T).T  # 计算世界坐标
    world_corners = world_corners[:, :3]  # 转换为非齐次坐标

    return world_corners


# 将世界坐标转换回图像坐标
def project_world_to_image(camera_matrix, dist_coeffs, rvec, tvec, world_points):
    image_points, _ = cv2.projectPoints(world_points, rvec, tvec, camera_matrix, dist_coeffs)
    return np.squeeze(image_points)


# 加载左右摄像头的标定结果
camera_matrix_left, dist_coeffs_left, rvec_left, tvec_left = load_calibration_results('left_camera_calibration.json')
camera_matrix_right, dist_coeffs_right, rvec_right, tvec_right = load_calibration_results(
    'right_camera_calibration.json')

# 图像尺寸
image_size = (477, 850)  # 高度, 宽度

# 获取左右摄像头图像四个角点在世界坐标系中的位置
world_corners_left = get_image_corners_in_world(camera_matrix_left, dist_coeffs_left, rvec_left, tvec_left, image_size)
world_corners_right = get_image_corners_in_world(camera_matrix_right, dist_coeffs_right, rvec_right, tvec_right,
                                                 image_size)

# 计算公共视场的边界（简单的方法：使用最小包围矩形）
world_corners_overlap = np.vstack((world_corners_left, world_corners_right))  # 合并左右图像的世界坐标
x_min, y_min = np.min(world_corners_overlap, axis=0)[:2]
x_max, y_max = np.max(world_corners_overlap, axis=0)[:2]

# 重叠区域的四个角点（在世界坐标系中）
world_corners_overlap_rect = np.array([[x_min, y_min, 0],
                                       [x_max, y_min, 0],
                                       [x_max, y_max, 0],
                                       [x_min, y_max, 0]], dtype=np.float32)

# 将重叠区域投影回左右图像中
image_corners_overlap_left = project_world_to_image(camera_matrix_left, dist_coeffs_left, rvec_left, tvec_left,
                                                    world_corners_overlap_rect)
image_corners_overlap_right = project_world_to_image(camera_matrix_right, dist_coeffs_right, rvec_right, tvec_right,
                                                     world_corners_overlap_rect)

# 在左右图像上绘制红框表示重叠区域
left_image = cv2.imread('left-850*477.png')
right_image = cv2.imread('right-850*477.png')

cv2.polylines(left_image, [np.int32(image_corners_overlap_left)], isClosed=True, color=(0, 0, 255), thickness=2)
cv2.polylines(right_image, [np.int32(image_corners_overlap_right)], isClosed=True, color=(0, 0, 255), thickness=2)

# 显示并保存结果
cv2.imshow("Left Image with Overlap", left_image)
cv2.imshow("Right Image with Overlap", right_image)
cv2.imwrite("left_with_overlap.png", left_image)
cv2.imwrite("right_with_overlap.png", right_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
