import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image_path = "img.png"
image = cv2.imread(image_path)

# 全局变量，用于存储四个角点
points = []

#Point selected: (354, 506)
#Point selected: (607, 506)
#Point selected: (666, 681)
#Point selected: (286, 686)

# 鼠标回调函数
def select_points(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("image", image)
        print(f"Point selected: ({x}, {y})")
        if len(points) == 4:
            cv2.destroyAllWindows()


# 显示图像并设置鼠标回调函数
cv2.imshow("image", image)
cv2.setMouseCallback("image", select_points)
cv2.waitKey(0)

# 检查是否选择了四个点
if len(points) == 4:
    chessboard_src = np.float32(points)
    print("Selected points:", chessboard_src)

    # 将浮点数转换为整数
    chessboard_src = chessboard_src.astype(int)

    # 提取左下角的棋盘图案
    x_min = min(point[0] for point in chessboard_src)
    x_max = max(point[0] for point in chessboard_src)
    y_min = min(point[1] for point in chessboard_src)
    y_max = max(point[1] for point in chessboard_src)

    chessboard = image[y_min:y_max, x_min:x_max]

    # 显示提取的棋盘图案
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(chessboard, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Extracted Chessboard Pattern')
    plt.show()

    # 定义棋盘的3D点
    chessboard_3d = np.zeros((9 * 9, 3), np.float32)
    chessboard_3d[:, :2] = np.mgrid[0:9, 0:9].T.reshape(-1, 2)
    chessboard_3d *= 10  # 每个格子10x10cm

    # 使用OpenCV的findHomography来找到变换矩阵
    H, status = cv2.findHomography(chessboard_3d[:, :2], chessboard_src)

    # 重建3D空间中的棋盘图案
    h, w = image.shape[:2]
    chessboard_reprojected = cv2.perspectiveTransform(np.array([chessboard_3d]), H)

    # 显示原始图像和重建的3D棋盘图案
    for point in chessboard_reprojected[0]:
        cv2.circle(image, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)

    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Reprojected Chessboard in 3D Space')
    plt.show()
else:
    print("Please select exactly four points.")
