import os
import torch
import numpy as np
from sklearn.decomposition import PCA

# 文件路径
feature_folder = '/scratchdata1/users/a1978372/Anxuan/clam/CLAM/Extracted_feature/pt_files'
save_folder = '/scratchdata1/users/a1978372/Anxuan/clam/CLAM/Processed_Data/features'

# 创建保存降维后特征的文件夹（如果不存在）
os.makedirs(save_folder, exist_ok=True)

def load_features_from_pt_file(file_path):
    """加载 .pt 文件中的特征矩阵"""
    features = torch.load(file_path)  # 直接加载
    return features.numpy()  # 转换成 numpy 数组

def reduce_dimensionality(features, n_components=256):
    """使用 PCA 进行降维"""
    pca = PCA(n_components=n_components, random_state=42)
    features_reduced = pca.fit_transform(features)
    return features_reduced, pca  # 返回降维后的特征和 PCA 模型（以防后续要用）

def save_reduced_features(features, save_path):
    """将降维后的特征保存到文件"""
    torch.save(torch.tensor(features, dtype=torch.float32), save_path)

def process_all_features(feature_folder, save_folder):
    """对目录下的所有 .pt 文件执行 PCA 降维并保存结果"""
    
    for file_name in os.listdir(feature_folder):
        if file_name.endswith('.pt'):
            file_path = os.path.join(feature_folder, file_name)
            
            # 加载特征
            features = load_features_from_pt_file(file_path)
            print(f"Processing {file_name}, original shape: {features.shape}")

            # 执行 PCA
            reduced_features, pca_model = reduce_dimensionality(features, n_components=256)
            print(f"Reduced shape: {reduced_features.shape}")

            # 保存降维后的特征
            save_path = os.path.join(save_folder, file_name)
            save_reduced_features(reduced_features, save_path)
            print(f"Saved reduced features for {file_name} to {save_path}")

# 执行 PCA 降维
process_all_features(feature_folder, save_folder)
