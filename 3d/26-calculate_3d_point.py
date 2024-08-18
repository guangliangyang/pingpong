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


# 将旋转向量和平移向量转换为投影矩阵
def get_projection_matrix(camera_matrix, rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)  # 将旋转向量转换为旋转矩阵
    Rt = np.hstack((R, tvec.reshape(-1, 1)))  # 构建 [R|t] 矩阵
    P = np.dot(camera_matrix, Rt)  # 计算投影矩阵 P = K * [R|t]
    return P


# 执行三角测量，计算3D点
def calculate_3d_point(proj_matrix_left, proj_matrix_right, point_2d_left, point_2d_right):
    points_2d_left = np.array([point_2d_left], dtype=np.float32).T
    points_2d_right = np.array([point_2d_right], dtype=np.float32).T
    point_4d_homogeneous = cv2.triangulatePoints(proj_matrix_left, proj_matrix_right, points_2d_left, points_2d_right)
    point_3d = point_4d_homogeneous[:3] / point_4d_homogeneous[3]  # 转换为非齐次坐标
    return point_3d.flatten()


# 鼠标回调函数，用于获取点击的2D点并在图像上绘制高亮点
def mouse_callback(event, x, y, flags, param):
    image, points = param
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        # 在图像上绘制高亮点
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # 红色高亮点
        cv2.imshow("Image", image)


# 加载左右摄像头的标定结果
camera_matrix_left, dist_coeffs_left, rvec_left, tvec_left = load_calibration_results('left_camera_calibration.json')
camera_matrix_right, dist_coeffs_right, rvec_right, tvec_right = load_calibration_results(
    'right_camera_calibration.json')

# 构建左右相机的投影矩阵
proj_matrix_left = get_projection_matrix(camera_matrix_left, rvec_left, tvec_left)
proj_matrix_right = get_projection_matrix(camera_matrix_right, rvec_right, tvec_right)

# 显示左图像并获取点击点
left_image = cv2.imread('left-850*477.png')
left_points = []
cv2.imshow("Image", left_image)
cv2.setMouseCallback("Image", mouse_callback, (left_image, left_points))

print("Click on the point in the left image, then press any key to continue...")
cv2.waitKey(0)

# 显示右图像并获取点击点
right_image = cv2.imread('right-850*477.png')
right_points = []
cv2.imshow("Image", right_image)
cv2.setMouseCallback("Image", mouse_callback, (right_image, right_points))

print("Click on the point in the right image, then press any key to continue...")
cv2.waitKey(0)

# 确保用户在每个图像中都点击了一个点
if len(left_points) == 1 and len(right_points) == 1:
    point_2d_left = left_points[0]
    point_2d_right = right_points[0]

    # 计算3D点的物理坐标
    point_3d = calculate_3d_point(proj_matrix_left, proj_matrix_right, point_2d_left, point_2d_right)
    print(f"The 3D coordinates of the point are: {point_3d}")
else:
    print("Error: You need to click on exactly one point in each image.")

cv2.destroyAllWindows()

'''
    
    代码说明：
get_projection_matrix 函数：将相机的内参矩阵、旋转向量和平移向量转换为相机的投影矩阵（P = K * [R|t]）。
calculate_3d_point 函数：使用 cv2.triangulatePoints 函数，通过左右摄像头的投影矩阵和2D点来计算3D坐标。这里使用齐次坐标系进行计算，最后将其转换为非齐次坐标。
输入左右图像中的2D点：手动输入左右图像中的2D点，它们代表同一个物理世界中的点。

    '''
