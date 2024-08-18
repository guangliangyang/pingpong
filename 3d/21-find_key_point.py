import cv2
import json
import os

# 加载图片
image_path = 'left-850*477.png'
image = cv2.imread(image_path)

# 用于保存点信息的列表
key_points = []

# JSON 文件路径
json_file_path = 'left-key-point.json'

# 检查 JSON 文件是否存在
if os.path.exists(json_file_path):
    # 加载已经存在的 JSON 文件
    with open(json_file_path, 'r') as json_file:
        key_points = json.load(json_file)

    # 在图片上绘制已经存在的点
    for point in key_points:
        x, y = point["image_coordinates"]
        point_name = point["point_name"]

        # 绘制圆圈和点名称
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(image, point_name, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    print(f"Loaded existing key points from {json_file_path}. Continue marking new points.")


# 回调函数，用于鼠标点击事件
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 获取点名称
        point_name = input("Enter the name of the point: ")

        # 获取3D坐标
        world3d_x = float(input(f"Enter the world 3D X coordinate for {point_name}: "))
        world3d_y = float(input(f"Enter the world 3D Y coordinate for {point_name}: "))
        world3d_z = float(input(f"Enter the world 3D Z coordinate for {point_name}: "))

        # 将点信息添加到列表中
        key_points.append({
            "point_name": point_name,
            "image_coordinates": [x, y],
            "world3d_coordinates": [world3d_x, world3d_y, world3d_z]
        })

        # 在图片上绘制圆圈标记点
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(image, point_name, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Image", image)


# 创建窗口并设置鼠标回调
cv2.imshow("Image", image)
cv2.setMouseCallback("Image", mouse_callback)

# 循环等待用户点击，按下 ESC 键退出并保存
print("Click on points on the image. For each point, enter the name and corresponding 3D coordinates.")
print("Press ESC to finish and save the points.")

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC 键的 ASCII 码是 27
        break

# 保存点数据到 JSON 文件
with open(json_file_path, 'w') as json_file:
    json.dump(key_points, json_file, indent=4)

print(f"Key points and 3D coordinates saved to {json_file_path}")

cv2.destroyAllWindows()

'''
在乒乓球台的场景中，通常会将坐标轴按照以下方式定义，以符合物理直观性和计算的方便性：

1. X 轴 (水平轴)
方向：沿着乒乓球台的宽度方向（左右方向）。
正方向：通常定义为从球台的一侧到另一侧。例如，从球台的左侧向右侧为 X 轴正方向。
2. Y 轴 (纵向轴)
方向：沿着乒乓球台的长度方向（前后方向）。
正方向：通常定义为从球台的一端到另一端。例如，从靠近摄像机的一端向远离摄像机的一端为 Y 轴正方向。
3. Z 轴 (高度轴)
方向：垂直于乒乓球台平面，即竖直方向。
正方向：通常定义为从球台的平面向上（即朝向天花板的方向为 Z 轴正方向）。
例子：
X 轴：如果你站在球台的一侧，X 轴的正方向将从你的左手边指向右手边。
Y 轴：Y 轴的正方向则是从你站立的位置（靠近自己的一端）指向球台的远端（对方的一端）。
Z 轴：Z 轴的正方向是从球台表面垂直向上。
坐标系的右手法则：
为了与计算机视觉和物理中常用的右手坐标系一致，可以使用右手法则来定义坐标系：

大拇指指向 X 轴正方向。
食指指向 Y 轴正方向。
中指指向 Z 轴正方向。
在这种定义下，乒乓球在球台上移动时，其 X 和 Y 坐标分别代表球在球台的左右和前后位置，而 Z 坐标则表示球的高度。


'''