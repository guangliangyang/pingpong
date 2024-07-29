import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# 示例数据
data = np.array([0, 1, 1, 2, 7, 8, 2, 3, 10, 11])
boundary_labels = np.array([0, 0, 0, 0, 1, 0, 0, 0, 1, 0])  # 边界标注
token_labels = np.array([0, 0, 0, 0, 1, 1, 0, 0, 2, 2])  # 类别标注

X = torch.tensor(data, dtype=torch.float32).unsqueeze(1).unsqueeze(0)
boundary_y = torch.tensor(boundary_labels, dtype=torch.float32).unsqueeze(1).unsqueeze(0)
token_y = torch.tensor(token_labels, dtype=torch.long).unsqueeze(0)

# TensorDataset 和 DataLoader
dataset = TensorDataset(X, boundary_y, token_y)
dataloader = DataLoader(dataset, batch_size=1)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, nhead, num_encoder_layers, dim_feedforward, num_classes):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(d_model=input_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                          dim_feedforward=dim_feedforward)
        self.fc_boundary = nn.Linear(input_dim, 1)
        self.fc_token = nn.Linear(input_dim, num_classes)

    def forward(self, src):
        transformer_out = self.transformer(src, src)
        boundary_out = torch.sigmoid(self.fc_boundary(transformer_out))
        token_out = self.fc_token(transformer_out)
        return boundary_out, token_out


# 模型参数
input_dim = 1
nhead = 1
num_encoder_layers = 2
dim_feedforward = 10
num_classes = 3

# 模型实例
model = TransformerModel(input_dim, nhead, num_encoder_layers, dim_feedforward, num_classes)

# 损失函数和优化器
criterion_boundary = nn.BCELoss()
criterion_token = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
model.train()
for epoch in range(100):
    for batch_X, batch_boundary_y, batch_token_y in dataloader:
        optimizer.zero_grad()
        boundary_output, token_output = model(batch_X)

        # 计算损失
        loss_boundary = criterion_boundary(boundary_output, batch_boundary_y)
        loss_token = criterion_token(token_output.squeeze(0), batch_token_y.squeeze(0))
        loss = loss_boundary + loss_token

        loss.backward()
        optimizer.step()


model.eval()
with torch.no_grad():
    boundary_predictions, token_predictions = model(X)
    predicted_boundaries = (boundary_predictions > 0.5).int().squeeze().numpy()
    predicted_tokens = torch.argmax(token_predictions, dim=-1).squeeze().numpy()

    print(predicted_boundaries)  # 输出预测的边界
    print(predicted_tokens)  # 输出预测的token类别
