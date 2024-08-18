import json
import numpy as np
import cv2


# 从JSON文件中读取标注好的数据
def load_keypoints(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    image_points = []
    world_points = []

    for point in data:
        image_points.append(point["image_coordinates"])
        world_points.append(point["world3d_coordinates"])

    return np.array(image_points, dtype=np.float32), np.array(world_points, dtype=np.float32)


# 将标定结果保存到JSON文件
def save_calibration_to_json(file_name, camera_matrix, dist_coeffs, rvecs, tvecs):
    calibration_data = {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
        "rvecs": [rvec.tolist() for rvec in rvecs],
        "tvecs": [tvec.tolist() for tvec in tvecs]
    }

    with open(file_name, 'w') as f:
        json.dump(calibration_data, f, indent=4)


# 加载左右摄像头的标注数据
left_image_points, left_world_points = load_keypoints('left-key-point.json')
right_image_points, right_world_points = load_keypoints('right-key-point.json')

# 假设世界坐标是相同的，因为标定板或物体相同
object_points = [left_world_points]  # 可以添加多个视角的世界点
left_image_points = [left_image_points]  # 同样可以添加多个视角的图像点
right_image_points = [right_image_points]


# 确保有足够的点对
assert len(left_image_points[0]) >= 8, "You need at least 8 point pairs for calibration."

# 摄像头图像的分辨率
image_size = (850, 477)

# 初始化摄像头内参矩阵，假设焦距为图像宽度，主点为图像中心
initial_camera_matrix = np.array([[image_size[0], 0, image_size[0] / 2],
                                  [0, image_size[0], image_size[1] / 2],
                                  [0, 0, 1]], dtype=np.float32)

# 假设无畸变
dist_coeffs = np.zeros(5, dtype=np.float32)

# 对左摄像头进行标定，使用初始内参矩阵
ret_left, camera_matrix_left, dist_coeffs_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
    object_points, left_image_points, image_size, initial_camera_matrix, dist_coeffs,
    flags=cv2.CALIB_USE_INTRINSIC_GUESS)

# 对右摄像头进行标定，使用初始内参矩阵
ret_right, camera_matrix_right, dist_coeffs_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
    object_points, right_image_points, image_size, initial_camera_matrix, dist_coeffs,
    flags=cv2.CALIB_USE_INTRINSIC_GUESS)

# 保存左摄像头标定结果到JSON文件
save_calibration_to_json('left_camera_calibration.json', camera_matrix_left, dist_coeffs_left, rvecs_left, tvecs_left)

# 保存右摄像头标定结果到JSON文件
save_calibration_to_json('right_camera_calibration.json', camera_matrix_right, dist_coeffs_right, rvecs_right,
                         tvecs_right)

print("Calibration results saved to JSON files.")

'''

这些假设是为了初始化摄像机的内参矩阵，但最终的参数会通过 cv2.calibrateCamera 函数根据输入的数据进行优化和调整。calibrateCamera 的作用就是在给定这些初始假设的基础上，通过最小化重投影误差来优化内参和外参。因此，你提供的初始内参矩阵只是算法优化的起点。

具体说明：
无畸变假设：

初始假设：我们在代码中假设畸变系数为零（dist_coeffs = np.zeros(5)），但 calibrateCamera 会根据输入的3D点和2D点对，自动估计真实的畸变系数。
自动调整：如果图像确实存在畸变，calibrateCamera 会优化这些系数，最终输出真实的畸变参数。你可以在标定结果中检查这些系数是否仍然接近于零。
焦距假设：

初始假设：焦距被初始化为图像宽度（例如，focal_length = image_size[0]），这是一个合理的初始估计值。
自动调整：calibrateCamera 会通过输入的数据优化焦距（即内参矩阵中的 [0,0] 和 [1,1] 元素）。最终的焦距值将基于实际的标定数据，可能与初始假设不同。
主点假设：

初始假设：主点被初始化为图像的中心（例如，cx = image_size[0] / 2，cy = image_size[1] / 2），这是常见的假设。
自动调整：calibrateCamera 同样会根据输入的数据优化主点的位置（即内参矩阵中的 [0,2] 和 [1,2] 元素），最终的主点位置可能会偏离图像中心，特别是在镜头发生轻微偏移的情况下。
总结：
初始假设：这些初始假设（无畸变、焦距为图像宽度、主点在图像中心）仅用于为 calibrateCamera 提供一个初始的内参矩阵。
自动调整：calibrateCamera 会根据输入的3D点和2D点对，自动优化这些参数，最终输出更符合实际情况的内参和外参。因此，尽管初始值很重要，但最终的参数是由标定过程决定的。

一般来说，至少要有10个以上的点对，越多越好
'''
