import torch
from torch import nn
from .TransformerEncoder import TransformerEncoder  # 确保TransformerEncoder.py文件在同一目录或者Python路径中, 加上.表示当前目录


class MultiTransformerClassifier(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, trans_output_dim, n_segments, num_classes, dropout=0.1):
        super(MultiTransformerClassifier, self).__init__()
        self.encoders = nn.ModuleList([
            TransformerEncoder(input_dim, model_dim, num_heads, num_layers, trans_output_dim, dropout)
            for _ in range(n_segments)
        ])
        # 多个Transformer输出融合后的维度可能需要调整，这里我们用一个全连接层进行维度匹配
        self.fc = nn.Linear(n_segments * trans_output_dim, num_classes)  # 假设每个编码器的输出也用于分类，所以是num_classes维

    def forward(self, x):
        # x: [batch_size, n_segments, sequence_length, input_dim]
        outputs = []

        for i, encoder in enumerate(self.encoders):
            segment_input = x[:, i, :, :]  # Shape: [batch_size, sequence_length, input_dim]
            output = encoder(segment_input)
            # print(f"Encoder {i} output shape: {output.shape}")
            outputs.append(output)  # 保存每个编码器的输出

        # Concatenate all encoder outputs
        concatenated = torch.cat(outputs, dim=1)  # Shape: [batch_size, n_segments * num_classes]
        logits = self.fc(concatenated)  # 最终分类
        return logits
