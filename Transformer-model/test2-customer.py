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
import logging

# 设置日志配置
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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
    logging.info(f"{folder_path} 文件夹已清空")

# MediaPipe 数据提取函数
def extract_mediapipe_data(video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    logging.info(f"视频总帧数: {frame_count}, FPS: {fps}")

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
            keypoints_data.append([(0, 0, 0)] * 33)  # MediaPipe 通常输出 33 个关键点

    cap.release()
    pose.close()

    return np.array(keypoints_data)

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        return x + self.pe[:x.size(0), :].to(x.device)

# Transformer 模型
class ActionTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, num_classes):
        super(ActionTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, src, src_key_padding_mask=None):
        logging.debug(f"src shape: {src.shape}")  # (batch_size, seq_len, input_dim)
        src = self.embedding(src)  # (batch_size, seq_len, d_model)
        logging.debug(f"After embedding src shape: {src.shape}")
        src = self.pos_encoder(src)  # (batch_size, seq_len, d_model)
        logging.debug(f"After positional encoding src shape: {src.shape}")
        src = src.permute(1, 0, 2)  # (seq_len, batch_size, d_model)
        logging.debug(f"After permute src shape: {src.shape}")
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        logging.debug(f"After transformer encoder output shape: {output.shape}")
        output = output.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        logging.debug(f"After permute output shape: {output.shape}")
        output = self.global_avg_pool(output).squeeze(2)  # (batch_size, d_model)
        logging.debug(f"After global_avg_pool output shape: {output.shape}")
        output = self.layer_norm(output)  # (batch_size, d_model)
        logging.debug(f"After layer_norm output shape: {output.shape}")
        output = F.relu(self.fc1(output))  # (batch_size, d_model // 2)
        logging.debug(f"After fc1 output shape: {output.shape}")
        output = self.fc2(output)  # (batch_size, num_classes)
        logging.debug(f"After fc2 output shape: {output.shape}")
        output = self.softmax(output)  # (batch_size, num_classes)
        logging.debug(f"After softmax output shape: {output.shape}")
        return output

def create_padding_mask(seq, pad_token=0):
    if seq.dim() == 2:
        # (batch_size, seq_len)
        mask = (seq == pad_token)
    elif seq.dim() == 3:
        # (batch_size, seq_len, input_dim)
        mask = (seq == pad_token).all(dim=-1)
    else:
        raise ValueError("Unsupported input dimensions")
    return mask


def read_label_data(csv_path):
    return pd.read_csv(csv_path)

def convert_labels_to_frames(label_data, total_frames, label_encoder):
    unknown_action = "unknown action"
    frame_labels = np.full(total_frames, label_encoder.transform([unknown_action])[0], dtype=int)
    start_label = label_encoder.transform(["start"])[0]
    end_label = label_encoder.transform(["end"])[0]

    for _, row in label_data.iterrows():
        start_frame = row['Start Frame']
        end_frame = row['End Frame']
        action_type = label_encoder.transform([row['Action Name']])[0]

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
    X = video_data.reshape(len(video_data), -1)  # Ensure (num_frames, 99)
    y = frame_labels
    return np.array(X), np.array(y)

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        inputs, labels = data.to(device), target.to(device)
        logging.debug(f"train input:{inputs}")
        logging.debug(f"train  labels:{labels}")
        logging.debug(f"Input shape: {inputs.shape}")  # (batch_size, seq_len, input_dim)
        src_key_padding_mask = create_padding_mask(inputs).to(device)  # (batch_size, seq_len)
        logging.debug(f"Padding mask shape: {src_key_padding_mask.shape}")

        optimizer.zero_grad()
        outputs = model(inputs, src_key_padding_mask=src_key_padding_mask)  # (batch_size, num_classes)
        outputs = outputs.view(labels.size(0), -1)  # 调整形状为 (batch_size, num_classes)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logging.debug(f"Validation Input shape: {inputs.shape}")  # (batch_size, seq_len, input_dim)
            src_key_padding_mask = create_padding_mask(inputs).to(device)  # (batch_size, seq_len)
            logging.debug(f"Padding mask shape: {src_key_padding_mask.shape}")

            outputs = model(inputs, src_key_padding_mask=src_key_padding_mask)  # (batch_size, num_classes)
            outputs = outputs.view(labels.size(0), -1)  # 调整形状为 (batch_size, num_classes)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    accuracy = correct / total
    return total_loss / len(val_loader), accuracy

def save_predictions(predictions, video_name, fps, label_encoder):
    output_path = os.path.join('predict', f'{video_name}_predict.csv')
    actions = detect_action_boundaries(predictions)
    data = []
    for start, end, action in actions:
        action = int(action)
        action_name = label_encoder.inverse_transform([action])[0]
        data.append({'Start Frame': start, 'End Frame': end, 'Action Name': action_name})
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)

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

def print_gpu_usage():
    if torch.cuda.is_available():
        logging.info(f"已分配的 GPU 内存: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        logging.info(f"GPU 内存缓存: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        logging.info(f"当前使用的 GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"GPU 索引: {torch.cuda.current_device()}")
    else:
        logging.info("没有可用的 GPU，使用 CPU")

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

    for video_file in video_files:
        video_name = os.path.splitext(video_file)[0]
        video_path = os.path.join(video_dir, video_file)
        label_path = os.path.join(label_dir, f'{video_name}_label.csv')

        if not os.path.exists(label_path):
            logging.warning(f"跳过视频 {video_name}，因为没有找到对应的标签文件")
            continue

        mediapipe_data_path = os.path.join(label_dir, f'{video_name}_full_mediapipe_data.csv')

        logging.info(f"处理视频: {video_name}")

        if os.path.exists(mediapipe_data_path):
            video_data = pd.read_csv(mediapipe_data_path).values.reshape(-1, 33, 3)
        else:
            video_data = extract_mediapipe_data(video_path)
            df = pd.DataFrame(video_data.reshape(video_data.shape[0], -1))
            df.to_csv(mediapipe_data_path, index=False)
            logging.info(f"MediaPipe 数据已保存到 {mediapipe_data_path}")

        label_data = read_label_data(label_path)
        all_action_types.extend(label_data['Action Name'].unique())

        all_video_data.append(video_data)
        all_frame_labels.append(label_data)
        processed_videos.append(video_name)

        logging.info(f"视频: {video_name}, 视频数据形状: {video_data.shape}, 标签数据形状: {label_data.shape}")

    if not all_video_data:
        logging.error("没有找到带有标签的视频，程序退出")
        exit()

    label_encoder.fit(all_action_types)

    X_all = []
    y_all = []
    for video_data, label_data, video_name in zip(all_video_data, all_frame_labels, processed_videos):
        frame_labels = convert_labels_to_frames(label_data, len(video_data), label_encoder)
        X, y = prepare_data(video_data, frame_labels)
        if X.size == 0 or y.size == 0:
            logging.warning(f"视频: {video_name}, 数据为空，跳过此视频")
            continue
        X_all.append(X)
        y_all.append(y)

        logging.info(f"视频: {video_name}, X 形状: {X.shape}, y 形状: {y.shape}")

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    logging.info(f"X_all 形状: {X_all.shape}, y_all 形状: {y_all.shape}")

    if len(X_all) == 0 or len(y_all) == 0:
        logging.error("没有有效的训练数据，程序退出")
        exit()

    X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

    num_classes = len(label_encoder.classes_)
    input_dim = 33 * 3
    d_model = 128
    model = ActionTransformer(input_dim=input_dim, d_model=d_model, nhead=8, num_layers=6, num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        logging.info(f"Epoch {epoch + 1}/{num_epochs}")
        logging.info(f"Train Loss: {train_loss:.4f}")
        logging.info(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            logging.info("保存最佳模型")

    model.load_state_dict(torch.load('best_model.pth'))
    logging.info("加载最佳模型")

    model.eval()

    overall_accuracy = []
    for video_name, video_data, label_data in zip(processed_videos, all_video_data, all_frame_labels):
        logging.info(f"\n预测视频: {video_name}")

        if video_data.size == 0:
            logging.warning(f"警告：视频 {video_name} 的数据为空，跳过此视频")
            continue

        if video_data.shape[0] < 30:
            logging.warning(f"警告：视频 {video_name} 的帧数不足（{video_data.shape[0]}帧），跳过此视频")
            continue

        video_data_flat = video_data.reshape(len(video_data), -1)
        video_tensor = torch.FloatTensor(video_data_flat).unsqueeze(0).to(device)
        src_key_padding_mask = create_padding_mask(video_tensor).to(device)
        logging.debug(f"Prediction Input shape: {video_tensor.shape}, Mask shape: {src_key_padding_mask.shape}")

        with torch.no_grad():
            output = model(video_tensor, src_key_padding_mask=src_key_padding_mask)
            frame_predictions = output.argmax(dim=1).cpu().numpy().flatten()

        video_path = os.path.join(video_dir, f"{video_name}.mp4")
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        save_predictions(frame_predictions, video_name, fps, label_encoder)

        true_labels = convert_labels_to_frames(label_data, len(video_data), label_encoder)
        accuracy = np.mean(np.array(frame_predictions) == true_labels)
        overall_accuracy.append(accuracy)
        logging.info(f"帧级别准确率: {accuracy:.2f}")

    logging.info("\n总体性能:")
    logging.info(f"平均帧级别准确率: {np.mean(overall_accuracy):.2f}")

    logging.info("\n动作类型映射:")
    for i, action_type in enumerate(label_encoder.classes_):
        logging.info(f"{action_type}: {i}")

    logging.info("\n所有带标签的视频处理完成")
