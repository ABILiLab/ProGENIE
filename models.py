import torch
import torch.nn as nn


class MLPModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPModel, self).__init__()
        
        # 先通过一个全连接层将输入数据转换为更小的维度
        self.fc1 = nn.Linear(input_dim, 512)  # 第一层
        self.fc2 = nn.Linear(512, 1024)       # 第二层
        self.fc3 = nn.Linear(1024, output_dim)  # 输出层
        
        # 使用ReLU激活函数
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # 假设 x 的形状为 [batch_size, 500, 256]
        x = x.view(x.size(0), -1)  # 展平为 [batch_size, 500 * 256] => [batch_size, 128000]
        x = self.relu(self.fc1(x))  # 通过第一层
        x = self.relu(self.fc2(x))  # 通过第二层
        x = self.fc3(x)             # 通过输出层，得到 [batch_size, 13591]
        return x
