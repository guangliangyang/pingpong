def draw_large_chessboard_pattern(frame,
                                  small_chessboard_size=(8, 8),
                                  small_square_size=10.0,
                                  large_square_width=100.0,
                                  large_square_height=75.0,
                                  vertical_offset=-40.0,
                                  num_large_squares_x=3,
                                  num_large_squares_y=6,
                                  show_overlay=False):
    """
    从视频帧中提取，计算相机标定参数，并在图像上绘制大棋盘格。

    参数:
    - frame: 视频帧
    - small_chessboard_size: 小棋盘格的内角点数量 (默认为 (8, 8))
    - small_square_size: 每个小格子的实际大小 (默认为 10.0 cm)
    - large_square_width: 每个大格子的实际宽度 (默认为 100.0 cm)
    - large_square_height: 每个大格子的实际高度 (默认为 75.0 cm)
    - vertical_offset: 原点在Y方向的偏移量 (默认为 -40.0 cm)
    - num_large_squares_x: 大棋盘格在X方向的数量 (默认为 3)
    - num_large_squares_y: 大棋盘格在Y方向的数量 (默认为 6)

    返回:
    - output_image: 带有大棋盘格的图像
    - chessboard_vertices: 四边形格子数组
    """
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
    objp[:, :2] = np.mgrid[0:small_chessboard_size[0], 0:small_chessboard_size[1]].T.reshape(-1, 2)
    objp *= small_square_size

    # 计算相机标定参数
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp], [corners], gray.shape[::-1], None, None)

    # 获取棋盘左下角的点作为原点
    left_bottom_corner = corners[-1][0]  # 选择左下角的角点作为原点

    # 创建大棋盘格的物理坐标，以左下角为原点，并向下移动30cm
    chessboard_physical_points = []
    for i in range(num_large_squares_y + 1):
        for j in range(num_large_squares_x + 1):
            chessboard_physical_points.append([j * large_square_width, -i * large_square_height - vertical_offset, 0])

    chessboard_physical_points = np.array(chessboard_physical_points, dtype=np.float32)

    # 将物理坐标转换为图像坐标
    def project_points(physical_points, rvec, tvec, mtx, dist):
        image_points, _ = cv2.projectPoints(physical_points, rvec, tvec, mtx, dist)
        return image_points.reshape(-1, 2)

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

    # 绘制大棋盘格
    output_image = frame.copy()
    if show_overlay:
        for vertices in chessboard_vertices:
            pts = np.array(vertices, dtype=np.int32)
            cv2.polylines(output_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # 找到最右侧中间大格子的右上角顶点的物理坐标
    middle_right_square_idx = (num_large_squares_y // 2) * num_large_squares_x + (num_large_squares_x - 1)
    right_top_vertex_idx = (num_large_squares_x + 1) * (middle_right_square_idx // num_large_squares_x) + (middle_right_square_idx % num_large_squares_x) + 1
    right_top_vertex_phys = chessboard_physical_points[right_top_vertex_idx]

    # 在该顶点上绘制一条垂直于棋盘的向上直线，长度为76 cm
    vertical_line_length = 76  # cm
    vertical_line_end_point_phys = right_top_vertex_phys + np.array([0, 0, vertical_line_length], dtype=np.float32)
    vertical_line_start_point_img, _ = cv2.projectPoints(right_top_vertex_phys.reshape(1, 1, 3), rvecs[0], tvecs[0], mtx, dist)
    vertical_line_end_point_img, _ = cv2.projectPoints(vertical_line_end_point_phys.reshape(1, 1, 3), rvecs[0], tvecs[0], mtx, dist)
    vertical_line_start_point_img = tuple(vertical_line_start_point_img.reshape(2).astype(int))
    vertical_line_end_point_img = tuple(vertical_line_end_point_img.reshape(2).astype(int))

    # 在图像上标注这些点的3D坐标
    cv2.putText(output_image, f"{right_top_vertex_phys}", (vertical_line_start_point_img[0], vertical_line_start_point_img[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(output_image, f"{vertical_line_end_point_phys}", (vertical_line_end_point_img[0], vertical_line_end_point_img[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    # 绘制垂直红线
    if (0 <= vertical_line_start_point_img[0] < output_image.shape[1] and
        0 <= vertical_line_start_point_img[1] < output_image.shape[0] and
        0 <= vertical_line_end_point_img[0] < output_image.shape[1] and
        0 <= vertical_line_end_point_img[1] < output_image.shape[0]):
        cv2.line(output_image, vertical_line_start_point_img, vertical_line_end_point_img, (0, 0, 255), 2)

    # 绘制水平线，长度为152 cm，与棋盘的Y方向平行，垂直线的终点为中点
    horizontal_line_length = 152  # cm
    horizontal_line_start_point_phys = right_top_vertex_phys + np.array([-horizontal_line_length / 2, 0, vertical_line_length], dtype=np.float32)
    horizontal_line_end_point_phys = right_top_vertex_phys + np.array([horizontal_line_length / 2, 0, vertical_line_length], dtype=np.float32)
    horizontal_line_start_point_img, _ = cv2.projectPoints(horizontal_line_start_point_phys.reshape(1, 1, 3), rvecs[0], tvecs[0], mtx, dist)
    horizontal_line_end_point_img, _ = cv2.projectPoints(horizontal_line_end_point_phys.reshape(1, 1, 3), rvecs[0], tvecs[0], mtx, dist)
    horizontal_line_start_point_img = tuple(horizontal_line_start_point_img.reshape(2).astype(int))
    horizontal_line_end_point_img = tuple(horizontal_line_end_point_img.reshape(2).astype(int))

    # 在图像上标注水平线端点的3D坐标
    cv2.putText(output_image, f"{horizontal_line_start_point_phys}", (horizontal_line_start_point_img[0], horizontal_line_start_point_img[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(output_image, f"{horizontal_line_end_point_phys}", (horizontal_line_end_point_img[0], horizontal_line_end_point_img[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    # 绘制水平红线
    if (0 <= horizontal_line_start_point_img[0] < output_image.shape[1] and
        0 <= horizontal_line_start_point_img[1] < output_image.shape[0] and
        0 <= horizontal_line_end_point_img[0] < output_image.shape[1] and
        0 <= horizontal_line_end_point_img[1] < output_image.shape[0]):
        cv2.line(output_image, horizontal_line_start_point_img, horizontal_line_end_point_img, (0, 0, 255), 2)

    # 归一化坐标
    height, width, _ = frame.shape
    normalized_chessboard_vertices = []
    for vertices in chessboard_vertices:
        normalized_vertices = [(pt[0] / width, pt[1] / height) for pt in vertices]
        normalized_chessboard_vertices.append(normalized_vertices)

    return output_image, normalized_chessboard_vertices, mtx, dist, rvecs[0], tvecs[0]
