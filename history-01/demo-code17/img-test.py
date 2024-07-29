import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image_path = "img.png"
image = cv2.imread(image_path)

# 使用固定坐标定义左下角棋盘的位置
chessboard_src = np.float32([
    [354, 506],  # 左上角
    [607, 506],  # 右上角
    [666, 681],  # 右下角
    [286, 686]   # 左下角
])

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

# 定义目标区域的四个角点
target_dst = np.float32([
    [445, 276],  # 左上角
    [364, 481],  # 左下角
    [1136, 479], # 右下角
    [783, 289]   # 右上角
])

# 计算目标区域的宽度和高度
width = int(np.linalg.norm(target_dst[0] - target_dst[3]))
height = int(np.linalg.norm(target_dst[0] - target_dst[1]))

# 生成目标区域的网格点
grid_x, grid_y = np.meshgrid(np.arange(0, width, chessboard.shape[1]), np.arange(0, height, chessboard.shape[0]))
grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

# 初始化结果图像
result = image.copy()

# 平铺棋盘图案
for point in grid_points:
    src_points = np.float32([
        [0, 0],
        [chessboard.shape[1] - 1, 0],
        [chessboard.shape[1] - 1, chessboard.shape[0] - 1],
        [0, chessboard.shape[0] - 1]
    ])

    dst_points = np.float32([
        point,
        [point[0] + chessboard.shape[1], point[1]],
        [point[0] + chessboard.shape[1], point[1] + chessboard.shape[0]],
        [point[0], point[1] + chessboard.shape[0]]
    ])

    H, _ = cv2.findHomography(src_points, dst_points)
    warped_chessboard = cv2.warpPerspective(chessboard, H, (image.shape[1], image.shape[0]))

    mask = np.zeros_like(result)
    cv2.fillConvexPoly(mask, np.int32(dst_points), (255, 255, 255))
    mask = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(result, mask)
    result = cv2.add(result, warped_chessboard)

# 显示结果图像
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Reprojected Chessboard Tiled in 3D Space')
plt.show()
