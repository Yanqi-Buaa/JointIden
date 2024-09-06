import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(p=0.3)  # 设置dropout概率
        self.relu = nn.ReLU()

    def forward(self, x):
        # 初始化隐层状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # 前向传播GRU
        x, _ = self.gru(x, h0)
        # 取最后一个时间步
        x = x[:, -1, :]
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
