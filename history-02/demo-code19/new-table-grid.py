def calculate_chessboard_data(frame,
                              small_chessboard_size=(8, 8),
                              small_square_size=10.0,
                              large_square_width=100.0,
                              large_square_height=75.0,
                              num_large_squares_x=3,
                              num_large_squares_y=6,
                              yolo_model_path='path/to/your/yolov10/model',
                              table_length=1.5,
                              table_height=0.76):
    """
    从视频帧中提取，计算相机标定参数，并计算大棋盘格的相关数据。

    参数:
    - frame: 视频帧
    - small_chessboard_size: 小棋盘格的内角点数量 (默认为 (8, 8))
    - small_square_size: 每个小格子的实际大小 (默认为 10.0 cm)
    - large_square_width: 每个大格子的实际宽度 (默认为 100.0 cm)
    - large_square_height: 每个大格子的实际高度 (默认为 75.0 cm)
    - num_large_squares_x: 大棋盘格在X方向的数量 (默认为 3)
    - num_large_squares_y: 大棋盘格在Y方向的数量 (默认为 6)
    - yolo_model_path: YOLO 模型路径
    - table_length: 乒乓球台长度（默认为1.5米）
    - table_height: 乒乓球台高度（默认为0.76米）

    返回:
    - chessboard_data: 包含计算结果的字典
    """
    import cv2
    import numpy as np
    from ultralytics import YOLOv10

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
    objp[:, :2] = np.mgrid[0:small_chessboard_size[0], 0:small_chessboard_size[1]].T.reshape(-1, 2)
    objp *= small_square_size

    # 计算相机标定参数
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp], [corners], gray.shape[::-1], None, None)

    # 加载 YOLO 模型
    model = YOLOv10(yolo_model_path)

    # 使用 YOLO 识别乒乓球台的角点
    results = model(frame)
    table_corners = []
    for result in results:
        for box in result.boxes:
            if box.cls == 15:  # 假设 15 是乒乓球台的标签
                table_corners.append((int(box.xyxy[0][0]), int(box.xyxy[0][1])))
                table_corners.append((int(box.xyxy[0][2]), int(box.xyxy[0][3])))

    if len(table_corners) < 2:
        raise ValueError("无法检测到足够的乒乓球台角点")

    # 使用相机内参和畸变系数对乒乓球台角点进行反投影，得到其3D坐标
    def backproject_2d_to_3d(point_2d, z, camera_matrix):
        # 将 2D 图像坐标转换为 3D 空间中的点，假设已知 z 坐标
        x = (point_2d[0] - camera_matrix[0, 2]) * z / camera_matrix[0, 0]
        y = (point_2d[1] - camera_matrix[1, 2]) * z / camera_matrix[1, 1]
        return np.array([x, y, z], dtype=np.float32)

    # 计算两个乒乓球台角点的3D坐标
    point_3d_1 = backproject_2d_to_3d(table_corners[0], table_height, mtx)
    point_3d_2 = backproject_2d_to_3d(table_corners[1], table_height, mtx)

    # 计算两个角点的中点作为新的原点
    origin_3d = (point_3d_1 + point_3d_2) / 2
    origin_3d[2] = 0  # 设置 z 坐标为 0

    # 创建大棋盘格的物理坐标，以新的原点为基准，并平行于乒乓球台的平面
    chessboard_physical_points = []
    for i in range(-num_large_squares_y // 2, num_large_squares_y // 2 + 1):
        for j in range(-num_large_squares_x, 1):
            x = origin_3d[0] + j * large_square_width
            y = origin_3d[1] + i * large_square_height
            chessboard_physical_points.append([x, y, 0])

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
            top_left = chessboard_image_points_px[(i + num_large_squares_y // 2) * (num_large_squares_x + 1) + j]
            top_right = chessboard_image_points_px[(i + num_large_squares_y // 2) * (num_large_squares_x + 1) + j + 1]
            bottom_right = chessboard_image_points_px[(i + 1 + num_large_squares_y // 2) * (num_large_squares_x + 1) + j + 1]
            bottom_left = chessboard_image_points_px[(i + 1 + num_large_squares_y // 2) * (num_large_squares_x + 1) + j]
            chessboard_vertices.append([top_left, top_right, bottom_right, bottom_left])

    # 归一化坐标
    normalized_chessboard_vertices = []
    for vertices in chessboard_vertices:
        normalized_vertices = [(pt[0] / width, pt[1] / height) for pt in vertices]
        normalized_chessboard_vertices.append(normalized_vertices)

    chessboard_data = {
        'chessboard_vertices': normalized_chessboard_vertices,
        'table_3d_points': [point_3d_1.tolist(), point_3d_2.tolist()],
        'origin_3d': origin_3d.tolist(),
        'mtx': mtx,
        'dist': dist,
        'rvecs': rvecs[0],
        'tvecs': tvecs[0]
    }

    return chessboard_data
