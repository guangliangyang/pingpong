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

# 定义棋盘的3D点
# 棋盘的每个格子是10x10厘米，棋盘为9x9格，总大小为90x90厘米
chessboard_3d = np.zeros((4, 3), np.float32)
chessboard_3d[:,:2] = np.array([
    [0, 0],        # 左上角
    [9*10, 0],     # 右上角
    [9*10, 9*10],  # 右下角
    [0, 9*10]      # 左下角
])

# 使用OpenCV的findHomography来找到变换矩阵
H, status = cv2.findHomography(chessboard_3d[:, :2], chessboard_src)

# 重建3D空间中的棋盘图案
# 生成棋盘格子的3D点，每个格子是10x10厘米
grid_3d = np.zeros((9*9, 3), np.float32)
grid_3d[:, :2] = np.mgrid[0:9, 0:9].T.reshape(-1, 2) * 10

# 对棋盘格子的3D点进行透视变换
chessboard_reprojected = cv2.perspectiveTransform(np.array([grid_3d[:, :2]]), H)

# 显示原始图像和重建的3D棋盘图案
for point in chessboard_reprojected[0]:
    cv2.circle(image, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)

plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Reprojected Chessboard in 3D Space')
plt.show()
