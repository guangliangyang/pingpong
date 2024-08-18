import cv2
import numpy as np

# Load the images
left_image = cv2.imread('left-850*477.png')
right_image = cv2.imread('right-850*477.png')

# Convert images to grayscale
left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

# Initialize the SIFT detector
sift = cv2.SIFT_create()

# Detect and compute keypoints and descriptors
keypoints_left, descriptors_left = sift.detectAndCompute(left_gray, None)
keypoints_right, descriptors_right = sift.detectAndCompute(right_gray, None)

# Use FLANN-based matcher to match keypoints
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors_left, descriptors_right, k=2)

# Apply the ratio test to filter good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Extract the matched keypoints' coordinates
left_points = np.float32([keypoints_left[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
right_points = np.float32([keypoints_right[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Compute the homography matrix using RANSAC
H, mask = cv2.findHomography(right_points, left_points, cv2.RANSAC, 5.0)

# Get the corners of the right image
h, w = right_image.shape[:2]
right_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

# Transform the corners of the right image to the left image's perspective
transformed_corners = cv2.perspectiveTransform(right_corners, H)

# 1. Draw the overlapping region as a red polygon on the left image
left_image_with_overlap = left_image.copy()
cv2.polylines(left_image_with_overlap, [np.int32(transformed_corners)], isClosed=True, color=(0, 0, 255), thickness=2)

# Save the result as "left_with_overlap.png"
cv2.imwrite("left_with_overlap.png", left_image_with_overlap)

# 2. Now, transform the corners of the left image back to the right image's perspective using the inverse homography
H_inv = np.linalg.inv(H)
left_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
transformed_corners_back = cv2.perspectiveTransform(left_corners, H_inv)

# Draw the overlapping region as a red polygon on the right image
right_image_with_overlap = right_image.copy()
cv2.polylines(right_image_with_overlap, [np.int32(transformed_corners_back)], isClosed=True, color=(0, 0, 255), thickness=2)

# Save the result as "right_with_overlap.png"
cv2.imwrite("right_with_overlap.png", right_image_with_overlap)

# Display results (Optional)
cv2.imshow("Left Image with Overlap", left_image_with_overlap)
cv2.imshow("Right Image with Overlap", right_image_with_overlap)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''
SIFT（Scale-Invariant Feature Transform，尺度不变特征变换）是一种用于图像特征提取的计算机视觉算法。它由 David Lowe 在 1999 年首次提出，并在 2004 年进一步完善。SIFT 能够检测和描述图像中的局部特征，并且具有尺度不变性和旋转不变性，即它可以在不同大小和旋转角度的图像中检测出相同的特征。

SIFT 的关键特点：
尺度不变性：SIFT 特征可以在不同尺度下检测到同一个物体的特征点，无论物体在图像中是放大还是缩小。
旋转不变性：SIFT 特征可以在图像旋转时保持不变，检测到相同的特征点。
鲁棒性：SIFT 对于图像中的噪声、光照变化、视角变化等具有较强的鲁棒性。
SIFT 的主要步骤：
尺度空间极值检测：在不同尺度下检测图像中的潜在特征点，通常通过高斯模糊和差分高斯函数（DoG）来实现。此步骤帮助找到潜在的特征点。
关键点定位：对每个潜在特征点进行精确定位，并去除低对比度点和边缘响应点，确保只有稳定的关键点被保留。
方向分配：基于关键点邻域的梯度方向分布，为每个关键点分配一个或多个方向，以实现旋转不变性。
关键点描述符生成：在关键点的邻域内生成梯度直方图，形成描述符，这些描述符用于匹配相似图像中的特征点。
特征点匹配：使用描述符匹配不同图像中的相似特征点，常用方法包括暴力匹配（Brute-Force Matcher）或 FLANN 匹配器。
SIFT 的应用：
图像匹配与拼接：SIFT 常用于将多张图像的特征点进行匹配，以拼接图像或进行场景重建。
物体识别：SIFT 能够在图像中检测和识别特定的物体，广泛应用于自动驾驶、机器人视觉等领域。
图像检索：通过提取图像特征，可以用于图像检索系统中，从大规模图像库中找到相似的图像。
示例：
SIFT 特征可以用于两个不同图像之间的匹配，如您在代码中看到的，通过提取特征点并进行匹配，我们可以计算图像之间的单应性变换（Homography），进而找到图像的重叠区域。

注意事项：
由于 SIFT 涉及专利问题，OpenCV 中的 SIFT 算法在标准安装中并不可用，只有在安装了 opencv-contrib 模块后才能使用。此外，SIFT 在处理大量特征点时计算开销较大，因此有时可能需要使用更轻量级的算法，如 ORB（Oriented FAST and Rotated BRIEF）。
'''
