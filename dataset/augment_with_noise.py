import cv2
import numpy as np
import os
import random
import shutil


def add_random_noise_objects(image, num_objects=5, object_size=(30, 30)):
    """
    在图片上添加随机干扰物体。

    :param image: 输入图像。
    :param num_objects: 要添加的随机物体数量。
    :param object_size: 随机物体的尺寸。
    :return: 添加了随机物体的图像。
    """
    h, w, _ = image.shape
    obj_w, obj_h = object_size
    for _ in range(num_objects):
        x = random.randint(0, w - obj_w)
        y = random.randint(0, h - obj_h)
        noise_object = np.random.randint(0, 256, (obj_h, obj_w, 3), dtype=np.uint8)
        image[y:y + obj_h, x:x + obj_w] = noise_object
    return image


def process_images_and_labels(image_input_dir, label_input_dir, image_output_dir, label_output_dir, num_copies=5,
                              num_objects=5, object_size=(30, 30)):
    """
    处理目录中的所有图片，添加随机干扰物体，并复制标注数据。

    :param image_input_dir: 包含原始图片的输入目录。
    :param label_input_dir: 包含原始标注数据的输入目录。
    :param image_output_dir: 存储处理后图片的输出目录。
    :param label_output_dir: 存储处理后标注数据的输出目录。
    :param num_copies: 每张图片生成的干扰图片数量。
    :param num_objects: 要添加的随机物体数量。
    :param object_size: 随机物体的尺寸。
    """
    print(f"Current working directory: {os.getcwd()}")
    if not os.path.exists(image_input_dir):
        print(f"Input image directory {image_input_dir} does not exist")
        return

    if not os.path.exists(label_input_dir):
        print(f"Input label directory {label_input_dir} does not exist")
        return

    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(label_output_dir, exist_ok=True)

    for filename in os.listdir(image_input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_input_dir, filename)
            label_filename = os.path.splitext(filename)[0] + ".txt"
            label_path = os.path.join(label_input_dir, label_filename)

            if not os.path.exists(label_path):
                print(f"Label file {label_path} does not exist for image {image_path}")
                continue

            image = cv2.imread(image_path)
            if image is None:
                print(f"Error reading image {image_path}")
                continue

            for i in range(num_copies):
                modified_image = add_random_noise_objects(image.copy(), num_objects, object_size)
                output_image_filename = f"{os.path.splitext(filename)[0]}_aug_{i:02d}.jpg"
                output_label_filename = f"{os.path.splitext(filename)[0]}_aug_{i:02d}.txt"

                output_image_path = os.path.join(image_output_dir, output_image_filename)
                output_label_path = os.path.join(label_output_dir, output_label_filename)

                cv2.imwrite(output_image_path, modified_image)
                shutil.copy(label_path, output_label_path)


import os
import sys

# Mount Google Drive
from google.colab import drive

drive.mount('/content/drive')

# 设置你的目录路径（包含空格）
my_path = '/content/drive/MyDrive/Colab Notebooks/robotic-vision/pingpong_table'

# 设置输入目录和输出目录
image_input_directory = f"{my_path}/dataset/images/train-original"
label_input_directory = f"{my_path}/dataset/labels/train-original"

image_output_directory = f"{my_path}/dataset/images/train"
label_output_directory = f"{my_path}/dataset/labels/train"

# 重命名目录
if os.path.exists(image_output_directory):
    os.rename(image_output_directory, image_input_directory)

if os.path.exists(label_output_directory):
    os.rename(label_output_directory, label_input_directory)

# 处理图片，添加随机干扰物体，并复制标注数据
process_images_and_labels(image_input_directory, label_input_directory, image_output_directory, label_output_directory,
                          num_copies=10, num_objects=5, object_size=(30, 30))
