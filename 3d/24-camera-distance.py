import json
import numpy as np
import cv2


# 从JSON文件中读取标定结果
def load_calibration_results(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    camera_matrix = np.array(data["camera_matrix"])
    dist_coeffs = np.array(data["dist_coeffs"])
    rvec = np.array(data["rvecs"])
    tvec = np.array(data["tvecs"])

    return camera_matrix, dist_coeffs, rvec, tvec


# 计算两个摄像机之间的距离
def calculate_camera_distance(tvec1, tvec2):
    # 计算两个平移向量之间的欧氏距离
    distance = np.linalg.norm(tvec1 - tvec2)
    return distance


# 加载左右摄像头的标定结果
camera_matrix_left, dist_coeffs_left, rvec_left, tvec_left = load_calibration_results('left_camera_calibration.json')
camera_matrix_right, dist_coeffs_right, rvec_right, tvec_right = load_calibration_results(
    'right_camera_calibration.json')

# 计算两个摄像机之间的距离
distance = calculate_camera_distance(tvec_left, tvec_right)

print(f"Distance between the two cameras: {distance} units")
