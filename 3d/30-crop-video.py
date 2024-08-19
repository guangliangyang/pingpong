import cv2
import json
import os
import numpy as np
from ultralytics import YOLO
import mediapipe as mp


# 加载3D坐标的JSON文件
def load_json(file_path):
    """加载 JSON 文件"""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return []


def save_json(file_path, data):
    """保存数据到 JSON 文件"""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def load_key_points_3d():
    """加载 3D 坐标的 JSON 文件"""
    with open('key-point-3d.json', 'r') as f:
        return json.load(f)


def get_3d_coordinates(key_points_3d, point_name):
    """根据名称查找对应的 3D 坐标"""
    for point in key_points_3d:
        if point["point_name"] == point_name:
            return point["world3d_coordinates"]
    return None


def draw_points(image, points):
    """在图像上绘制加载的点"""
    for point in points:
        x, y = point["image_coordinates"]
        point_name = point["point_name"]
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        cv2.putText(image, point_name, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def mouse_callback(event, x, y, flags, param):
    """鼠标点击事件回调函数，用于标注点"""
    image, points_list, window_name, key_points_3d = param
    if event == cv2.EVENT_LBUTTONDOWN:
        point_name = input(f"Enter the name of the point for {window_name}: ")
        world3d_coordinates = get_3d_coordinates(key_points_3d, point_name)
        if world3d_coordinates is None:
            print(f"Point name '{point_name}' not found in key-point-3d.json")
            return

        points_list.append({
            "point_name": point_name,
            "image_coordinates": [x, y],
            "world3d_coordinates": world3d_coordinates
        })

        # 在图像上绘制圆圈和点名称
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(image, point_name, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow(window_name, image)


def mark_image_points(image, image_points, window_name, key_points_3d):
    """在图像上进行标注操作"""
    print(f"Start marking points on the {window_name.lower()} image.")
    draw_points(image, image_points)
    cv2.imshow(window_name, image)
    cv2.setMouseCallback(window_name, mouse_callback, [image, image_points, window_name, key_points_3d])

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC 键的 ASCII 码是 27
            break

    save_json(f'{window_name.lower()}_image_points.json', image_points)
    print(f"{window_name} image points saved to {window_name.lower()}_image_points.json")
    cv2.destroyAllWindows()


def calibrate_camera(image_points, world_points, image_size):
    """使用标注点对摄像头进行标定"""
    object_points = [world_points]
    image_points = [image_points]

    initial_camera_matrix = np.array([[image_size[0], 0, image_size[0] / 2],
                                      [0, image_size[0], image_size[1] / 2],
                                      [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros(5, dtype=np.float32)

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, initial_camera_matrix, dist_coeffs,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS)

    return camera_matrix, dist_coeffs, rvecs, tvecs


def save_calibration(file_name, camera_matrix, dist_coeffs, rvecs, tvecs):
    """将标定结果保存到 JSON 文件"""
    calibration_data = {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
        "rvecs": [rvec.tolist() for rvec in rvecs],
        "tvecs": [tvec.tolist() for tvec in tvecs]
    }
    save_json(file_name, calibration_data)
    print(f"Calibration results saved to {file_name}")


def detect_pingpong_ball(model, image):
    """使用 YOLO 模型检测乒乓球"""
    results = model(image)
    for result in results[0].boxes:
        if int(result.cls) == 3:  # 假设类别标签为 3 表示乒乓球
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return image


def process_video_for_detection(video_path, model, y_min, y_max, x_min, x_max, mid_point, pose, proj_matrix_left, proj_matrix_right):
    """处理视频并进行乒乓球检测和人体骨架绘制"""
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cropped_frame = frame[y_min:y_max+1, x_min:x_max+1]
        left_image = cropped_frame[:, :mid_point]
        right_image = cropped_frame[:, mid_point:]

        # YOLO 检测乒乓球
        left_image = detect_pingpong_ball(model, left_image)
        right_image = detect_pingpong_ball(model, right_image)

        # 使用 Mediapipe 绘制人体骨架
        left_image, left_keypoints = draw_skeleton_on_image(left_image, pose)
        right_image, right_keypoints = draw_skeleton_on_image(right_image, pose)

        # 计算关节点6和32之间的3D距离
        if left_keypoints and right_keypoints:
            distance = calculate_joint_distance(proj_matrix_left, proj_matrix_right, left_keypoints, right_keypoints)
            print(f"3D distance between joint 6 and joint 32: {distance:.2f} units")

        # 显示检测结果
        cv2.imshow("Left Image", left_image)
        cv2.imshow("Right Image", right_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def draw_skeleton_on_image(image, pose):
    """使用 Mediapipe 在图像上绘制人体骨架并返回关键点"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    keypoints = []

    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
            mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)
        )
        keypoints = [(lm.x, lm.y) for lm in results.pose_landmarks.landmark]

    return image, keypoints


def calculate_joint_distance(proj_matrix_left, proj_matrix_right, keypoints_left, keypoints_right):
    """计算关节点6和32之间的3D距离"""
    point_2d_left_6 = keypoints_left[6]
    point_2d_right_6 = keypoints_right[6]
    point_2d_left_32 = keypoints_left[32]
    point_2d_right_32 = keypoints_right[32]

    # 通过三角测量计算3D位置
    point_3d_6 = calculate_3d_point(proj_matrix_left, proj_matrix_right, point_2d_left_6, point_2d_right_6)
    point_3d_32 = calculate_3d_point(proj_matrix_left, proj_matrix_right, point_2d_left_32, point_2d_right_32)

    print(f"Joint 6 (3D): {point_3d_6}")
    print(f"Joint 32 (3D): {point_3d_32}")

    # 计算3D距离
    distance = np.linalg.norm(point_3d_6 - point_3d_32)
    return distance


def calculate_3d_point(proj_matrix_left, proj_matrix_right, point_2d_left, point_2d_right):
    """执行三角测量，计算3D点"""
    points_2d_left = np.array([point_2d_left], dtype=np.float32).T
    points_2d_right = np.array([point_2d_right], dtype=np.float32).T
    point_4d_homogeneous = cv2.triangulatePoints(proj_matrix_left, proj_matrix_right, points_2d_left, points_2d_right)
    point_3d = point_4d_homogeneous[:3] / point_4d_homogeneous[3]  # 转换为非齐次坐标
    return point_3d.flatten()


def get_projection_matrix(camera_matrix, rvec, tvec):
    """将旋转向量和平移向量转换为投影矩阵"""
    R, _ = cv2.Rodrigues(rvec)  # 将旋转向量转换为旋转矩阵
    Rt = np.hstack((R, tvec.reshape(-1, 1)))  # 构建 [R|t] 矩阵
    P = np.dot(camera_matrix, Rt)  # 计算投影矩阵 P = K * [R|t]
    return P


def load_keypoints(json_path):
    """从 JSON 文件中读取标注好的数据"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    image_points = []
    world_points = []

    for point in data:
        image_points.append(point["image_coordinates"])
        world_points.append(point["world3d_coordinates"])

    return np.array(image_points, dtype=np.float32), np.array(world_points, dtype=np.float32)


def main():
    key_points_3d = load_key_points_3d()

    # 加载左右图像标注数据
    left_image_points = load_json('left_image_points.json')
    right_image_points = load_json('right_image_points.json')

    # 读取视频并提取第一帧
    video_path = 'speed-01.mov'
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("无法读取视频文件")
        exit()

    # 图像预处理
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary_frame = cv2.threshold(gray_frame, 1, 255, cv2.THRESH_BINARY)
    vertical_projection = np.sum(binary_frame, axis=1)
    horizontal_projection = np.sum(binary_frame, axis=0)
    y_min = np.where(vertical_projection > 0)[0][0]
    y_max = np.where(vertical_projection > 0)[0][-1]
    x_min = np.where(horizontal_projection > 0)[0][0]
    x_max = np.where(horizontal_projection > 0)[0][-1]
    cropped_frame = frame[y_min:y_max+1, x_min:x_max+1]
    height, width, _ = cropped_frame.shape
    mid_point = width // 2
    left_image = cropped_frame[:, :mid_point]
    right_image = cropped_frame[:, mid_point:]

    # 标注左图像点
    mark_image_points(left_image, left_image_points, "Left Image", key_points_3d)

    # 标注右图像点
    mark_image_points(right_image, right_image_points, "Right Image", key_points_3d)

    # 加载标注数据
    left_image_points, left_world_points = load_keypoints('left_image_points.json')
    right_image_points, right_world_points = load_keypoints('right_image_points.json')

    # 摄像头标定
    image_size = (left_image.shape[1], left_image.shape[0])
    left_camera_matrix, left_dist_coeffs, left_rvecs, left_tvecs = calibrate_camera(left_image_points, left_world_points, image_size)
    right_camera_matrix, right_dist_coeffs, right_rvecs, right_tvecs = calibrate_camera(right_image_points, right_world_points, image_size)

    save_calibration('left_camera_calibration.json', left_camera_matrix, left_dist_coeffs, left_rvecs, left_tvecs)
    save_calibration('right_camera_calibration.json', right_camera_matrix, right_dist_coeffs, right_rvecs, right_tvecs)

    # 构建投影矩阵
    proj_matrix_left = get_projection_matrix(left_camera_matrix, left_rvecs[0], left_tvecs[0])
    proj_matrix_right = get_projection_matrix(right_camera_matrix, right_rvecs[0], right_tvecs[0])

    # 乒乓球检测和人体骨架绘制
    model_path = "/Users/andy/workspace/AUT-PY/Table_tennis_demo_zh/best_bak.pt"
    model = YOLO(model_path)

    with mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        process_video_for_detection(video_path, model, y_min, y_max, x_min, x_max, mid_point, pose, proj_matrix_left, proj_matrix_right)


if __name__ == "__main__":
    main()
