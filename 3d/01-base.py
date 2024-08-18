import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建3D图形对象
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 定义3D点的坐标
x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 6, 7, 8, 9])
z = np.array([9, 8, 7, 6, 5])

# 绘制3D点
ax.scatter(x, y, z, c='r', marker='o')

# 设置坐标轴标签
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_ylabel('Z Label')

# 设置标题
ax.set_title('3D Point Visualization')

# 绘制3D线
ax.plot(x, y, z, label='3D Line')
ax.legend()

# 创建一个曲面网格
X = np.linspace(-5, 5, 100)
Y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(X, Y)
Z = np.sin(np.sqrt(X**2 + Y**2))

# 绘制3D曲面
ax.plot_surface(X, Y, Z, cmap='viridis')


# 显示图形
plt.show()

