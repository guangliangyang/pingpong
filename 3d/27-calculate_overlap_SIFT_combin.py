import cv2
import numpy as np

# 加载图像
left_image = cv2.imread('left-850*477.png')
right_image = cv2.imread('right-850*477.png')

# 转换为灰度图像
left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

# 初始化 SIFT 检测器
sift = cv2.SIFT_create()

# 检测并计算特征点和描述符
keypoints_left, descriptors_left = sift.detectAndCompute(left_gray, None)
keypoints_right, descriptors_right = sift.detectAndCompute(right_gray, None)

# 使用 FLANN 匹配器匹配特征点
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors_left, descriptors_right, k=2)

# 应用比值测试筛选匹配点
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# 排除不合理的匹配点：计算连线的平行度和长度差异
filtered_matches = []
angle_threshold = 10  # 允许的最大角度差（单位：度）
length_ratio_threshold = 0.5  # 允许的长度比例差

for match in good_matches:
    # 获取左图像和右图像中的匹配点坐标
    pt_left = np.array(keypoints_left[match.queryIdx].pt)
    pt_right = np.array(keypoints_right[match.trainIdx].pt)

    # 计算两点的连线方向（使用 atan2 计算角度）
    delta_left = pt_left - np.array([left_image.shape[1] / 2, left_image.shape[0] / 2])
    delta_right = pt_right - np.array([right_image.shape[1] / 2, right_image.shape[0] / 2])

    angle_left = np.degrees(np.arctan2(delta_left[1], delta_left[0]))
    angle_right = np.degrees(np.arctan2(delta_right[1], delta_right[0]))

    # 计算连线的角度差
    angle_diff = abs(angle_left - angle_right)

    # 将角度差规范化到 [0, 180] 范围
    if angle_diff > 180:
        angle_diff = 360 - angle_diff

    # 计算两点之间的距离（长度）
    length_left = np.linalg.norm(delta_left)
    length_right = np.linalg.norm(delta_right)

    # 计算长度的比例差
    length_ratio = abs(length_left - length_right) / max(length_left, length_right)

    # 判断是否符合平行度和长度差异的要求
    if angle_diff <= angle_threshold and length_ratio <= length_ratio_threshold:
        filtered_matches.append(match)

# 提取过滤后的匹配点坐标
if len(filtered_matches) >= 4:  # 至少需要 4 个匹配点
    left_pts = np.float32([keypoints_left[m.queryIdx].pt for m in filtered_matches])
    right_pts = np.float32([keypoints_right[m.trainIdx].pt for m in filtered_matches])

    # 计算左图像中匹配点的边界框
    x_left_min, y_left_min = np.int32(left_pts.min(axis=0))
    x_left_max, y_left_max = np.int32(left_pts.max(axis=0))

    # 计算右图像中匹配点的边界框
    x_right_min, y_right_min = np.int32(right_pts.min(axis=0))
    x_right_max, y_right_max = np.int32(right_pts.max(axis=0))

    # 在左图像上绘制红框
    left_image_with_box = left_image.copy()
    cv2.rectangle(left_image_with_box, (x_left_min, y_left_min), (x_left_max, y_left_max), (0, 0, 255), 2)

    # 在右图像上绘制红框
    right_image_with_box = right_image.copy()
    cv2.rectangle(right_image_with_box, (x_right_min, y_right_min), (x_right_max, y_right_max), (0, 0, 255), 2)

    # 将两幅图像水平拼接在一起，方便可视化
    combined_image = np.hstack((left_image_with_box, right_image_with_box))

    # 绘制匹配的点并连接
    for i in range(len(filtered_matches)):
        pt_left = tuple(np.int32(left_pts[i]))
        pt_right = tuple(np.int32(right_pts[i] + np.array([left_image.shape[1], 0])))  # 右图像在水平拼接后的位置
        cv2.circle(combined_image, pt_left, 5, (0, 255, 0), -1)
        cv2.circle(combined_image, pt_right, 5, (0, 255, 0), -1)
        cv2.line(combined_image, pt_left, pt_right, (255, 0, 0), 1)

    # 显示并保存结果
    cv2.imshow("Combined Image with Matched Points and Overlap Boxes", combined_image)
    cv2.imwrite("combined_with_overlap_boxes.png", combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("匹配点不足，无法计算重叠范围。")

