import cv2
import numpy as np

def calculate_chessboard_data(frame,
                              small_chessboard_size=(3, 3),
                              small_square_size=15.0,
                              large_square_width=100.0,
                              large_square_height=75.0,
                              vertical_offset=0.0,
                              num_large_squares_x=3,
                              num_large_squares_y=6):
    """
    从视频帧中提取，计算相机标定参数，并计算大棋盘格的相关数据。

    参数:
    - frame: 视频帧
    - small_chessboard_size: 小棋盘格的内角点数量 (默认为 (3, 3))
    - small_square_size: 每个小格子的实际大小 (默认为 15.0 cm)
    - large_square_width: 每个大格子的实际宽度 (默认为 100.0 cm)
    - large_square_height: 每个大格子的实际高度 (默认为 75.0 cm)
    - vertical_offset: 原点在Y方向的偏移量 (默认为 0.0 cm)
    - num_large_squares_x: 大棋盘格在X方向的数量 (默认为 3)
    - num_large_squares_y: 大棋盘格在Y方向的数量 (默认为 6)

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
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp], [corners], gray.shape[::-1], initial_camera_matrix, dist_coeffs, flags=cv2.CALIB_USE_INTRINSIC_GUESS)

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

    # 获取小棋盘上沿的两点
    horizontal_start_point_phys = objp[0]
    horizontal_end_point_phys = objp[small_chessboard_size[0] - 1]

    # 转换为图像坐标
    def project_points(physical_points, rvec, tvec, mtx, dist):
        image_points, _ = cv2.projectPoints(physical_points, rvec, tvec, mtx, dist)
        return image_points.reshape(-1, 2)

    horizontal_start_point_img, _ = cv2.projectPoints(horizontal_start_point_phys.reshape(1, 1, 3), rvecs[0], tvecs[0], mtx, dist)
    horizontal_end_point_img, _ = cv2.projectPoints(horizontal_end_point_phys.reshape(1, 1, 3), rvecs[0], tvecs[0], mtx, dist)
    horizontal_start_point_img = tuple(map(int, horizontal_start_point_img.reshape(2)))
    horizontal_end_point_img = tuple(map(int, horizontal_end_point_img.reshape(2)))

    # 获取大棋盘格在地面上的物理坐标
    chessboard_physical_points = []
    for i in range(num_large_squares_y + 1):
        for j in range(num_large_squares_x + 1):
            point = np.array([j * large_square_width, vertical_offset, -i * large_square_height])
            chessboard_physical_points.append(point)

    chessboard_physical_points = np.array(chessboard_physical_points, dtype=np.float32)

    # 将物理坐标转换为图像坐标
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

    # 计算新的原点：向下移动两个单元格，并在小棋盘平面上向左移动四个单元格
    new_origin_phys = objp[0] - np.array([4 * small_square_size, 0, 2 * small_square_size], dtype=np.float32)
    new_origin_img, _ = cv2.projectPoints(new_origin_phys.reshape(1, 1, 3), rvecs[0], tvecs[0], mtx, dist)
    new_origin_img = tuple(map(int, new_origin_img.reshape(2)))

    # 计算垂直线的3D物理坐标
    vertical_end_point_phys = new_origin_phys + normal_vector * 76  # 向上76 cm

    vertical_line_1_end_phys = new_origin_phys + normal_vector * 76
    vertical_line_2_end_phys = horizontal_end_point_phys + normal_vector * 76

    vertical_end_point_img, _ = cv2.projectPoints(vertical_end_point_phys.reshape(1, 1, 3), rvecs[0], tvecs[0], mtx, dist)
    vertical_line_1_end_img, _ = cv2.projectPoints(vertical_line_1_end_phys.reshape(1, 1, 3), rvecs[0], tvecs[0], mtx, dist)
    vertical_line_2_end_img, _ = cv2.projectPoints(vertical_line_2_end_phys.reshape(1, 1, 3), rvecs[0], tvecs[0], mtx, dist)
    vertical_end_point_img = tuple(map(int, vertical_end_point_img.reshape(2)))
    vertical_line_1_end_img = tuple(map(int, vertical_line_1_end_img.reshape(2)))
    vertical_line_2_end_img = tuple(map(int, vertical_line_2_end_img.reshape(2)))

    # 归一化坐标
    normalized_chessboard_vertices = []
    for vertices in chessboard_vertices:
        normalized_vertices = [(pt[0] / width, pt[1] / height) for pt in vertices]
        normalized_chessboard_vertices.append(normalized_vertices)

    # 将 new_origin_img 也进行归一化
    new_origin_img_normalized = (new_origin_img[0] / width, new_origin_img[1] / height)

    chessboard_data = {
        'chessboard_vertices': normalized_chessboard_vertices,
        'right_top_vertex_img': new_origin_img_normalized,
        'vertical_end_point_img': (vertical_end_point_img[0] / width, vertical_end_point_img[1] / height),
        'horizontal_start_point_img': (horizontal_start_point_img[0] / width, horizontal_start_point_img[1] / height),
        'horizontal_end_point_img': (horizontal_end_point_img[0] / width, horizontal_end_point_img[1] / height),
        'vertical_line_1_end_img': (vertical_line_1_end_img[0] / width, vertical_line_1_end_img[1] / height),
        'vertical_line_2_end_img': (vertical_line_2_end_img[0] / width, vertical_line_2_end_img[1] / height),
        'mtx': mtx,
        'dist': dist,
        'rvecs': rvecs[0],
        'tvecs': tvecs[0],
        'small_chessboard_corners': [(int(c[0][0]), int(c[0][1])) for c in corners],  # 小棋盘的内角点坐标
        'new_origin_img': new_origin_img_normalized  # 新的原点图像坐标
    }

    return chessboard_data

def draw_chessboard_on_frame(frame, chessboard_data, show_overlay=False):
    """
    在当前帧上绘制大棋盘格和相关连线。

    参数:
    - frame: 当前帧
    - chessboard_data: 包含大棋盘格相关数据的字典
    - show_overlay: 是否显示棋盘格的覆盖层
    """
    output_image = frame.copy()
    height, width, _ = frame.shape

    if not show_overlay:
        return output_image

    for vertices in chessboard_data['chessboard_vertices']:
        pts = np.array([(int(pt[0] * width), int(pt[1] * height)) for pt in vertices], dtype=np.int32)
        cv2.polylines(output_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    if 'right_top_vertex_img' in chessboard_data and 'vertical_end_point_img' in chessboard_data:
        right_top_vertex_img = (int(chessboard_data['right_top_vertex_img'][0] * width),
                                int(chessboard_data['right_top_vertex_img'][1] * height))
        vertical_end_point_img = (int(chessboard_data['vertical_end_point_img'][0] * width),
                                  int(chessboard_data['vertical_end_point_img'][1] * height))
        cv2.line(output_image, right_top_vertex_img, vertical_end_point_img, (255, 255, 255), 2)

    if 'horizontal_start_point_img' in chessboard_data and 'horizontal_end_point_img' in chessboard_data:
        horizontal_start_point_img = (int(chessboard_data['horizontal_start_point_img'][0] * width),
                                      int(chessboard_data['horizontal_start_point_img'][1] * height))
        horizontal_end_point_img = (int(chessboard_data['horizontal_end_point_img'][0] * width),
                                    int(chessboard_data['horizontal_end_point_img'][1] * height))
        cv2.line(output_image, horizontal_start_point_img, horizontal_end_point_img, (255, 255, 255), 2)

    if 'vertical_line_1_end_img' in chessboard_data and 'vertical_line_2_end_img' in chessboard_data:
        vertical_line_1_end_img = (int(chessboard_data['vertical_line_1_end_img'][0] * width),
                                   int(chessboard_data['vertical_line_1_end_img'][1] * height))
        vertical_line_2_end_img = (int(chessboard_data['vertical_line_2_end_img'][0] * width),
                                   int(chessboard_data['vertical_line_2_end_img'][1] * height))
        cv2.line(output_image, horizontal_start_point_img, vertical_line_1_end_img, (255, 255, 255), 2)
        cv2.line(output_image, horizontal_end_point_img, vertical_line_2_end_img, (255, 255, 255), 2)

    # 高亮显示新的原点
    if 'new_origin_img' in chessboard_data:
        new_origin_img = (int(chessboard_data['new_origin_img'][0] * width),
                          int(chessboard_data['new_origin_img'][1] * height))
        cv2.circle(output_image, new_origin_img, 10, (0, 0, 255), -1)

    return output_image

# 处理01.png的示例
image_path = '01.png'
frame = cv2.imread(image_path)
try:
    chessboard_data = calculate_chessboard_data(frame, small_chessboard_size=(3, 3), small_square_size=15.0, large_square_width=100.0, large_square_height=75.0, vertical_offset=0.0, num_large_squares_x=3, num_large_squares_y=6)
    output_image = draw_chessboard_on_frame(frame, chessboard_data, show_overlay=True)

    # 显示结果
    cv2.imshow('Output Image', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except ValueError as e:
    print(e)
