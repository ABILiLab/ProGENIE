import torch
from torch.utils.data import Dataset
import pandas as pd
import os

class CustomDataset(Dataset):
    def __init__(self, features_dir, labels_csv):
        """
        初始化自定义数据集类，加载特征和标签。
        
        features_dir: 特征文件夹路径
        labels_csv: 标签CSV文件路径
        """
        # 加载标签
        self.labels_df = pd.read_csv(labels_csv, index_col=0)
        
        # 获取特征文件的排序顺序
        self.features_dir = features_dir
        self.feature_files = sorted(os.listdir(features_dir))  # 按照字母顺序排序文件名
        self.sample_ids = [f.split('.')[0] for f in self.feature_files]  # 提取文件名前缀作为样本ID
        
        # 确保标签和样本ID匹配
        self.labels_df = self.labels_df.loc[self.sample_ids]
        
    def __len__(self):
        """返回数据集的大小"""
        return len(self.feature_files)
    
    def __getitem__(self, idx):
        """根据索引获取一个样本的特征和标签"""
        # 获取特征文件路径
        feature_file = self.feature_files[idx]
        feature_path = os.path.join(self.features_dir, feature_file)
        
        # 加载特征
        features = torch.load(feature_path)  # 这里假设文件格式是.pt
        
        # 获取标签
        sample_id = self.sample_ids[idx]
        label = self.labels_df.loc[sample_id].values  # 标签是一个数组
        
        return features, torch.tensor(label, dtype=torch.float32)




features_dir = "/scratchdata1/users/a1978372/Anxuan/clam/CLAM/Processed_Data/features2_padded"
labels_csv = "/scratchdata1/users/a1978372/Anxuan/clam/CLAM/Processed_Data/labels/model_rnaseq_clean_sorted.csv"

# 创建 Dataset 和 DataLoader
dataset = CustomDataset(features_dir, labels_csv)

print(f"Dataset size: {len(dataset)}")  # 输出数据集大小

