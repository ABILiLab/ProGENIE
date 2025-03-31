import torch
import os
import numpy as np
from tqdm import tqdm  # 导入tqdm进度条

# 定义原始和目标文件夹路径
source_dir = '/scratchdata1/users/a1978372/Anxuan/clam/CLAM/Processed_Data/features2'
target_dir = '/scratchdata1/users/a1978372/Anxuan/clam/CLAM/Processed_Data/features2_padded'

# 创建目标文件夹（如果不存在）
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 目标形状
target_shape = (500, 256)

# 获取文件列表并使用tqdm包装
file_list = [f for f in os.listdir(source_dir) if f.endswith('.pt')]

# 使用 tqdm 显示进度条
for filename in tqdm(file_list, desc="Processing files", unit="file"):
    file_path = os.path.join(source_dir, filename)
    
    # 加载特征矩阵
    features = torch.load(file_path)
    
    # 如果 features 是 numpy.ndarray，先转换成 Tensor
    if isinstance(features, np.ndarray):
        features = torch.tensor(features, dtype=torch.float32)

    # 检查当前矩阵的形状
    current_shape = features.shape

    # 如果形状不符合目标，进行填充
    if current_shape != target_shape:
        # 计算需要的填充量
        padding = [
            (0, target_shape[0] - current_shape[0]),  # 补充到5000行
            (0, target_shape[1] - current_shape[1])   # 补充到256列
        ]
        # 使用 pad 函数进行填充
        padded_features = torch.nn.functional.pad(
            features, 
            (0, padding[1][1], 0, padding[0][1]),  # `pad` 需要按照 (最后维度, 第一维度) 的顺序
            mode='constant', 
            value=0
        )
    else:
        padded_features = features

    # 保存填充后的特征矩阵
    target_file_path = os.path.join(target_dir, filename)
    torch.save(padded_features, target_file_path)
