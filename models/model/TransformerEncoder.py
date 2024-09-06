import torch
import torch.nn as nn
import numpy as np


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.model_dim = model_dim
        self.embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = self.create_positional_encoding(5000, model_dim)  # Assuming max length of 5000
        encoder_layers = nn.TransformerEncoderLayer(model_dim, num_heads, model_dim * 4, dropout, batch_first=True)  # model_dim * 4是FFN中间层的大小，这个设计是基于Transformer原始论文的常见实践
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(model_dim, output_dim)

    def create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 保持 batch 维度
        return pe

    def forward(self, src):
        src = self.embedding(src) * np.sqrt(self.model_dim)
        src = src + self.positional_encoding[:, :src.size(1), :].to(src.device)  # 确保位置编码在相同的设备
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)  # 对时间维度进行全局平均池化
        output = self.fc(output)
        return output
