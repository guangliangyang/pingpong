import cv2
import json
import os

# 加载图片
image_path = 'right-850*477.png'
image = cv2.imread(image_path)

# 用于保存点信息的列表
key_points = []

# JSON 文件路径
json_file_path = 'right-key-point.json'

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
