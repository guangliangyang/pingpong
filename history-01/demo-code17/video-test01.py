import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# 定义棋盘格尺寸，根据实际情况调整
chessboard_size = (8, 8)  # 9x9的格子有8x8的内角点
square_size = 10.0  # 每个格子的实际大小为10cm

# 准备世界坐标系中的3D点
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# 用于存储所有图像的3D点和2D点
objpoints = []  # 3D点
imgpoints = []  # 2D点

# 从视频中提取帧
video_path = os.path.join('..', 'mp4', '01.mov')  # 替换为实际的视频文件名
cap = cv2.VideoCapture(video_path)

frame_interval = 30  # 每隔30帧提取一个
frame_count = 0
max_frames = 10  # 仅提取10张图片
extracted_frames = 0
frames = []

while cap.isOpened() and extracted_frames < max_frames:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)

        # 找到棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            extracted_frames += 1

            # 可视化角点
            cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)
            cv2.imshow('Corners', frame)
            cv2.waitKey(500)

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

if extracted_frames < max_frames:
    print("提取的图片数量不足10张，请选择一个更长的视频或降低frame_interval。")
else:
    # 进行摄像机标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # 畸变校正和保存帧
    corrected_frames = []
    for frame in frames:
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        # 畸变校正
        dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)

        # 裁剪图像
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
        corrected_frames.append(dst)

    # 保存畸变校正后的图像
    output_folder = 'corrected_frames'
    os.makedirs(output_folder, exist_ok=True)
    for i, corrected_frame in enumerate(corrected_frames):
        corrected_frame_path = os.path.join(output_folder, f"corrected_frame_{i}.jpg")
        cv2.imwrite(corrected_frame_path, corrected_frame)

    # 特征检测和匹配
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(corrected_frames[0], None)
    keypoints2, descriptors2 = orb.detectAndCompute(corrected_frames[1], None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    img_matches = cv2.drawMatches(corrected_frames[0], keypoints1, corrected_frames[1], keypoints2, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('Matches', img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 提取匹配点
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # 计算基础矩阵
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

    # 计算本质矩阵
    E = mtx.T @ F @ mtx

    # 从本质矩阵中恢复姿态
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, mtx)

    # 三角测量重建3D点
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((R, t))

    pts1 = cv2.undistortPoints(pts1, mtx, dist)
    pts2 = cv2.undistortPoints(pts2, mtx, dist)

    points_4D_hom = cv2.triangulatePoints(P1, P2, pts1, pts2)
    points_4D = points_4D_hom / points_4D_hom[3]

    # 打印3D点
    print("3D points:\n", points_4D[:3].T)

    # 可视化3D点
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_4D[0], points_4D[1], points_4D[2], s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
