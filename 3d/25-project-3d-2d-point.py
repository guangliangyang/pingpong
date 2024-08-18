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


# 将3D点投影到图像平面上
def project_point_onto_image(camera_matrix, dist_coeffs, rvec, tvec, point_3d):
    point_3d = np.array([point_3d], dtype=np.float32)
    image_points, _ = cv2.projectPoints(point_3d, rvec, tvec, camera_matrix, dist_coeffs)
    return image_points[0][0]


# 在图像上绘制点
def draw_point_on_image(image_path, point_2d, output_path):
    image = cv2.imread(image_path)
    point_2d = tuple(int(x) for x in point_2d)
    cv2.circle(image, point_2d, 5, (0, 0, 255), -1)
    cv2.imwrite(output_path, image)


# 加载左右摄像头的标定结果
camera_matrix_left, dist_coeffs_left, rvec_left, tvec_left = load_calibration_results('left_camera_calibration.json')
camera_matrix_right, dist_coeffs_right, rvec_right, tvec_right = load_calibration_results(
    'right_camera_calibration.json')

# 3D点 (76.25, 68.5, 0)
point_3d = (76.25, 68.5, 0)

# 将3D点投影到左摄像头图像上
point_2d_left = project_point_onto_image(camera_matrix_left, dist_coeffs_left, rvec_left, tvec_left, point_3d)

# 将3D点投影到右摄像头图像上
point_2d_right = project_point_onto_image(camera_matrix_right, dist_coeffs_right, rvec_right, tvec_right, point_3d)

# 在左图像上绘制点
draw_point_on_image('left-850*477.png', point_2d_left, 'left_with_projected_point.png')

# 在右图像上绘制点
draw_point_on_image('right-850*477.png', point_2d_right, 'right_with_projected_point.png')

print(f"Left camera 2D point: {point_2d_left}")
print(f"Right camera 2D point: {point_2d_right}")
