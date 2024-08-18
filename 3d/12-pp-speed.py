import cv2
import numpy as np
import matplotlib.pyplot as plt

# 初始化一些变量
positions_3d = []  # 存储每一帧乒乓球的3D坐标
timestamps = []    # 存储每一帧的时间戳

# 假设摄像机的内外参已经计算好，省略之前的PnP步骤
# P1 和 P2 是两个摄像机的投影矩阵

# 读取视频或连续帧
cap1 = cv2.VideoCapture('camera1_video.mp4')
cap2 = cv2.VideoCapture('camera2_video.mp4')

while cap1.isOpened() and cap2.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break

    # YOLO检测乒乓球在两个图像中的位置（假设已知检测代码）
    ball_2d_cam1 = np.array([[u_ball_cam1, v_ball_cam1]], dtype=np.float32).T
    ball_2d_cam2 = np.array([[u_ball_cam2, v_ball_cam2]], dtype=np.float32).T

    # 使用三角测量法恢复乒乓球的3D坐标
    ball_4d_homogeneous = cv2.triangulatePoints(P1, P2, ball_2d_cam1, ball_2d_cam2)
    ball_3d = ball_4d_homogeneous[:3] / ball_4d_homogeneous[3]

    # 存储3D坐标和时间戳
    positions_3d.append(ball_3d.flatten())
    timestamps.append(cv2.getTickCount() / cv2.getTickFrequency())

cap1.release()
cap2.release()

# 计算乒乓球的速度
velocities = []
for i in range(1, len(positions_3d)):
    delta_pos = np.linalg.norm(positions_3d[i] - positions_3d[i - 1])  # 计算两帧之间的3D距离
    delta_time = timestamps[i] - timestamps[i - 1]  # 计算时间差
    velocity = delta_pos / delta_time  # 计算速度
    velocities.append(velocity)

# 打印平均速度
average_velocity = np.mean(velocities)
print(f"Average velocity of the ping-pong ball: {average_velocity:.2f} m/s")

# 3D轨迹可视化
positions_3d = np.array(positions_3d)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(positions_3d[:, 0], positions_3d[:, 1], positions_3d[:, 2], label='Ping-pong ball trajectory')
ax.scatter(positions_3d[:, 0], positions_3d[:, 1], positions_3d[:, 2], color='red')  # 标记每一个点
ax.set_xlabel('X axis (meters)')
ax.set_ylabel('Y axis (meters)')
ax.set_zlabel('Z axis (meters)')
ax.legend()
plt.show()

'''
步骤1：捕捉多帧图像，计算乒乓球在每一帧的3D位置
利用前述的方法，在连续的多帧图像中检测乒乓球，并计算每一帧的乒乓球3D坐标。
保存每一帧的3D坐标和时间戳，用于后续速度计算。
步骤2：计算乒乓球的速度
速度可以通过两帧之间的3D坐标变化和时间差来计算。利用公式 
𝑣
=
Δ
𝑑
Δ
𝑡
v= 
Δt
Δd
​
 ，即速度等于两帧之间的距离变化除以时间差。
步骤3：绘制乒乓球的3D轨迹
使用Matplotlib或其他3D绘图库来可视化乒乓球的运动轨迹

------------------------------
利用YOLO模型检测乒乓球台的关键点，例如台面四个角点和球网的两端点。
对每个摄像机，获取这些关键点的2D像素坐标。
使用PnP算法计算每个摄像机的内外参：

已知乒乓球台关键点的3D坐标（例如通过物理测量得到）。
使用YOLO检测到的2D像素坐标，通过OpenCV的solvePnP函数计算每个摄像机的内外参。
这些内外参将用于3D空间建模。
将两个摄像机的坐标系对齐到同一个世界坐标系：

选择一个摄像机的坐标系作为世界坐标系，或者将乒乓球台中心定义为世界坐标系的原点。
使用两个摄像机之间的相对位姿关系来转换它们的坐标系。
利用YOLO检测到的乒乓球位置计算3D坐标：

在两个摄像机图像中使用YOLO检测乒乓球的2D像素坐标。
使用两个摄像机的投影矩阵和YOLO检测到的2D点，通过三角测量法计算乒乓球的3D坐标。
------------------------------

'''