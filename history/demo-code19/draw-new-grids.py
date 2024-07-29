import cv2
import numpy as np


def calculate_chessboard_data(frame,
                              small_chessboard_size=(3, 3),
                              small_square_size=15.0):
    """
    从视频帧中提取，计算小棋盘的角点，并计算新的原点。

    参数:
    - frame: 视频帧
    - small_chessboard_size: 小棋盘格的内角点数量 (默认为 (3, 3))
    - small_square_size: 每个小格子的实际大小 (默认为 15.0 cm)

    返回:
    - chessboard_data: 包含计算结果的字典
    """

    def clamp_point(point, width, height):
        """确保点在图像范围内"""
        x, y = point
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        return x, y

    # 确保 frame 非空
    if frame is None:
        raise ValueError("输入帧为空，请检查图像路径是否正确")

    height, width, _ = frame.shape

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
    for i in range(small_chessboard_size[1]):
        for j in range(small_chessboard_size[0]):
            objp[i * small_chessboard_size[0] + j] = [j * small_square_size, 0, (i + 2) * small_square_size]

    # 设置初始内在参数矩阵
    focal_length = max(width, height)
    center = (width / 2, height / 2)
    initial_camera_matrix = np.array([[focal_length, 0, center[0]],
                                      [0, focal_length, center[1]],
                                      [0, 0, 1]], dtype=float)

    dist_coeffs = np.zeros((4, 1))  # 假设畸变系数初始为0

    # 计算相机标定参数
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp], [corners], gray.shape[::-1], initial_camera_matrix,
                                                       dist_coeffs, flags=cv2.CALIB_USE_INTRINSIC_GUESS)

    # 获取棋盘第一个点作为原点
    origin_corner = corners[0][0]

    # 计算小棋盘平面的法向量
    p1 = objp[0]
    p2 = objp[1]
    p3 = objp[small_chessboard_size[0]]
    v1 = p2 - p1
    v2 = p3 - p1
    normal_vector = np.cross(v1, v2)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)  # 归一化法向量

    # 计算新的原点：在小棋盘的平面上向左和向下移动
    new_origin_phys = objp[0] - np.array([-3 * small_square_size, 0, 2 * small_square_size], dtype=np.float32)
    new_origin_img, _ = cv2.projectPoints(new_origin_phys.reshape(1, 1, 3), rvecs[0], tvecs[0], mtx, dist)
    new_origin_img = tuple(map(int, new_origin_img.reshape(2)))

    # 计算法向量终点
    normal_vector_end_phys = new_origin_phys + normal_vector * 50  # 法向量长度为50
    normal_vector_end_img, _ = cv2.projectPoints(normal_vector_end_phys.reshape(1, 1, 3), rvecs[0], tvecs[0], mtx, dist)
    normal_vector_end_img = tuple(map(int, normal_vector_end_img.reshape(2)))

    # 归一化坐标
    new_origin_img_normalized = (new_origin_img[0] / width, new_origin_img[1] / height)
    normal_vector_end_img_normalized = (normal_vector_end_img[0] / width, normal_vector_end_img[1] / height)

    chessboard_data = {
        'small_chessboard_corners': [(int(c[0][0]), int(c[0][1])) for c in corners],  # 小棋盘的内角点坐标
        'new_origin_img': new_origin_img_normalized,  # 新的原点图像坐标
        'normal_vector_end_img': normal_vector_end_img_normalized  # 法向量终点图像坐标
    }

    return chessboard_data


def draw_chessboard_on_frame(frame, chessboard_data, show_overlay=False):
    """
    在当前帧上绘制小棋盘格和新的原点。

    参数:
    - frame: 当前帧
    - chessboard_data: 包含小棋盘格相关数据的字典
    - show_overlay: 是否显示棋盘格的覆盖层
    """
    output_image = frame.copy()
    height, width, _ = frame.shape

    if not show_overlay:
        return output_image

    # 高亮显示小棋盘的所有角点
    if 'small_chessboard_corners' in chessboard_data:
        for corner in chessboard_data['small_chessboard_corners']:
            cv2.circle(output_image, corner, 5, (0, 255, 255), -1)

    # 高亮显示新的原点
    if 'new_origin_img' in chessboard_data:
        new_origin_img = (int(chessboard_data['new_origin_img'][0] * width),
                          int(chessboard_data['new_origin_img'][1] * height))
        cv2.circle(output_image, new_origin_img, 10, (0, 0, 255), -1)

    # 绘制法向量
    if 'normal_vector_end_img' in chessboard_data:
        normal_vector_end_img = (int(chessboard_data['normal_vector_end_img'][0] * width),
                                 int(chessboard_data['normal_vector_end_img'][1] * height))
        cv2.line(output_image, new_origin_img, normal_vector_end_img, (255, 0, 0), 2)

    return output_image


# 处理01.png的示例
image_path = '01.png'
frame = cv2.imread(image_path)
try:
    chessboard_data = calculate_chessboard_data(frame, small_chessboard_size=(3, 3), small_square_size=15.0)
    output_image = draw_chessboard_on_frame(frame, chessboard_data, show_overlay=True)

    # 显示结果
    cv2.imshow('Output Image', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except ValueError as e:
    print(e)
