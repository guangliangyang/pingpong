import os
import shutil
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import math


# 清空预测文件夹
def clear_predict_folder(folder_path='predict'):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    else:
        os.makedirs(folder_path)
    print(f"{folder_path} 文件夹已清空")


# MediaPipe 数据提取函数
def extract_mediapipe_data(video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"视频总帧数: {frame_count}, FPS: {fps}")

    keypoints_data = []

    for _ in tqdm(range(frame_count), desc="提取MediaPipe数据"):
        success, image = cap.read()
        if not success:
            break

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            keypoints = [(lm.x, lm.y, lm.z) for lm in landmarks]
            keypoints_data.append(keypoints)
        else:
            # 如果没有检测到姿势，添加一个全零的帧
            keypoints_data.append([(0, 0, 0)] * 33)  # MediaPipe 通常输出 33 个关键点

    cap.release()
    pose.close()

    return np.array(keypoints_data)


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        pe = self.pe[:x.size(0), :].to(x.device)
        return x + pe


# Transformer 模型
class ActionTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, src):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        # 添加全局池化
        output = output.mean(dim=1)  # 或者 output = output[:, -1, :]
        return self.classifier(output)


def detect_action_boundaries(frame_predictions, min_action_length=1):
    actions = []
    current_action = frame_predictions[0]
    start = 0

    for i, pred in enumerate(frame_predictions[1:], 1):
        if pred != current_action:
            if i - start >= min_action_length:
                actions.append((start, i - 1, current_action))
            start = i
            current_action = pred

    if len(frame_predictions) - start >= min_action_length:
        actions.append((start, len(frame_predictions) - 1, current_action))

    return actions


def read_label_data(csv_path):
    df = pd.read_csv(csv_path)
    return df


def convert_labels_to_frames(label_data, total_frames, label_encoder):
    unknown_action = "unknown action"
    frame_labels = np.full(total_frames, label_encoder.transform([unknown_action])[0], dtype=int)
    start_label = label_encoder.transform(["start"])[0]
    end_label = label_encoder.transform(["end"])[0]

    for _, row in label_data.iterrows():
        start_frame = row['Start Frame']
        end_frame = row['End Frame']
        action_type = label_encoder.transform([row['Action Name']])[0]

        # 标记 start 和 end 及其前后2帧
        start_range = range(max(start_frame - 2, 0), min(start_frame + 3, total_frames))
        end_range = range(max(end_frame - 2, 0), min(end_frame + 3, total_frames))

        for frame in start_range:
            frame_labels[frame] = start_label
        for frame in end_range:
            frame_labels[frame] = end_label

        if end_frame - start_frame > 1:
            frame_labels[start_frame + 3:end_frame - 2] = action_type

    return frame_labels


def prepare_data(video_data, frame_labels):
    X = video_data.reshape(len(video_data), -1)  # 保持数据的二维形状
    y = frame_labels
    return np.array(X), np.array(y)


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        inputs, labels = data.to(device), target.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        # 确保 outputs 和 labels 的维度匹配
        outputs = outputs.view(labels.size(0), -1)  # 调整输出形状
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:  # 每 100 个批次打印一次
            print(f'Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
            print_gpu_usage()

        total_loss += loss.item()
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            print(f"Inputs shape: {inputs.shape}, Labels shape: {labels.shape}")
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            outputs = outputs.view(-1, outputs.size(-1))  # 调整输出形状
            labels = labels.view(-1)  # 调整标签形状
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = correct / total
    return total_loss / len(val_loader), accuracy


def save_mediapipe_data(data, file_path):
    df = pd.DataFrame(data.reshape(data.shape[0], -1))
    df.to_csv(file_path, index=False)
    print(f"MediaPipe 数据已保存到 {file_path}")


def load_mediapipe_data(file_path):
    df = pd.read_csv(file_path)
    data = df.values.reshape(-1, 33, 3)
    print(f"MediaPipe 数据已从 {file_path} 加载")
    return data


def save_predictions(predictions, video_name, fps, label_encoder):
    output_path = os.path.join('predict', f'{video_name}_predict.csv')

    actions = detect_action_boundaries(predictions)

    # 添加调试信息
    print(f"预测结果 (前50帧): {predictions[:50]}")
    print(f"检测到的动作边界: {actions}")

    data = []
    for start, end, action in actions:
        action_name = label_encoder.inverse_transform([action])[0]
        data.append({
            'Start Frame': start,
            'End Frame': end,
            'Action Name': action_name
        })

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"预测结果已保存到 {output_path}")


def print_gpu_usage():
    if torch.cuda.is_available():
        print(f"已分配的 GPU 内存: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        print(f"GPU 内存缓存: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print(f"当前使用的 GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU 索引: {torch.cuda.current_device()}")
    else:
        print("没有可用的 GPU，使用 CPU")

    # 清空预测文件夹
    clear_predict_folder('predict')

    # 获取当前目录下mp4文件夹中的所有视频文件
    video_dir = os.path.join(os.getcwd(), 'video')
    label_dir = os.path.join(os.getcwd(), 'labels')
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mov', '.mp4'))]

    all_video_data = []
    all_frame_labels = []
    label_encoder = LabelEncoder()
    all_action_types = ["start", "end", "unknown action"]
    processed_videos = []

    # 首先，收集所有有标签的视频数据
    for video_file in video_files:
        video_name = os.path.splitext(video_file)[0]
        video_path = os.path.join(video_dir, video_file)
        label_path = os.path.join(label_dir, f'{video_name}_label.csv')

        # 检查是否存在对应的标签文件
        if not os.path.exists(label_path):
            print(f"跳过视频 {video_name}，因为没有找到对应的标签文件")
            continue

        mediapipe_data_path = os.path.join(label_dir, f'{video_name}_full_mediapipe_data.csv')

        print(f"处理视频: {video_name}")

        # 加载或提取MediaPipe数据
        if os.path.exists(mediapipe_data_path):
            video_data = load_mediapipe_data(mediapipe_data_path)
        else:
            video_data = extract_mediapipe_data(video_path)
            save_mediapipe_data(video_data, mediapipe_data_path)

        # 读取标注数据
        label_data = read_label_data(label_path)
        all_action_types.extend(label_data['Action Name'].unique())

        # 存储视频数据和标签
        all_video_data.append(video_data)
        all_frame_labels.append(label_data)
        processed_videos.append(video_name)

        # 添加调试信息
        print(f"视频: {video_name}, 视频数据形状: {video_data.shape}, 标签数据形状: {label_data.shape}")

    if not all_video_data:
        print("没有找到带有标签的视频，程序退出")
        exit()

    # 创建标签编码器
    label_encoder.fit(all_action_types)

    # 处理所有数据
    X_all = []
    y_all = []
    for video_data, label_data, video_name in zip(all_video_data, all_frame_labels, processed_videos):
        frame_labels = convert_labels_to_frames(label_data, len(video_data), label_encoder)
        X, y = prepare_data(video_data, frame_labels)
        if X.size == 0 or y.size == 0:
            print(f"视频: {video_name}, 数据为空，跳过此视频")
            continue
        X_all.append(X)
        y_all.append(y)

        # 添加调试信息
        print(f"视频: {video_name}, X 形状: {X.shape}, y 形状: {y.shape}")

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    # 添加调试信息
    print(f"X_all 形状: {X_all.shape}, y_all 形状: {y_all.shape}")

    # 划分训练集和验证集
    if len(X_all) == 0 or len(y_all) == 0:
        print("没有有效的训练数据，程序退出")
        exit()

    X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

    # 初始化模型
    num_classes = len(label_encoder.classes_)
    input_dim = 33 * 3  # 每帧的关键点数乘以每个关键点的坐标数
    d_model = 128  # 隐藏层维度，可以调整
    model = ActionTransformer(input_dim=input_dim, d_model=d_model, nhead=8, num_layers=6, num_classes=num_classes)
    print(model)
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    num_epochs = 50
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("保存最佳模型")

        print()

    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    print("加载最佳模型")

    model.eval()

    # 对每个处理过的视频进行预测和评估
    overall_accuracy = []
    for video_name, video_data, label_data in zip(processed_videos, all_video_data, all_frame_labels):
        print(f"\n预测视频: {video_name}")

        if video_data.size == 0:
            print(f"警告：视频 {video_name} 的数据为空，跳过此视频")
            continue

        # 检查视频帧数是否足够
        if video_data.shape[0] < 30:  # 假设至少需要30帧
            print(f"警告：视频 {video_name} 的帧数不足（{video_data.shape[0]}帧），跳过此视频")
            continue

        # 进行预测
        video_data_flat = video_data.reshape(len(video_data), -1)
        video_tensor = torch.FloatTensor(video_data_flat).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(video_tensor)
            frame_predictions = output.argmax(dim=2).cpu().numpy().flatten()

        # 保存预测结果
        video_path = os.path.join(video_dir, f"{video_name}.mp4")
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        save_predictions(frame_predictions, video_name, fps, label_encoder)

        # 评估模型性能
        true_labels = convert_labels_to_frames(label_data, len(video_data), label_encoder)
        accuracy = np.mean(np.array(frame_predictions) == true_labels)
        overall_accuracy.append(accuracy)
        print(f"帧级别准确率: {accuracy:.2f}")

    print("\n总体性能:")
    print(f"平均帧级别准确率: {np.mean(overall_accuracy):.2f}")

    # 输出标签编码映射
    print("\n动作类型映射:")
    for i, action_type in enumerate(label_encoder.classes_):
        print(f"{action_type}: {i}")

    print("\n所有带标签的视频处理完成")
