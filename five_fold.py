import os
import torch
from torch.utils.data import Subset
from sklearn.model_selection import KFold
from dataset import CustomDataset  # 确保你已正确定义 CustomDataset

# 数据路径
features_dir = '/scratchdata1/users/a1978372/Anxuan/clam/CLAM/Processed_Data/features2_padded'
labels_csv = '/scratchdata1/users/a1978372/Anxuan/clam/CLAM/Processed_Data/labels/model_rnaseq_clean_sorted.csv'

# 加载数据集
dataset = CustomDataset(features_dir, labels_csv)
dataset_size = len(dataset)
print(f"Dataset size: {dataset_size}")

# 5折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 创建保存数据集的目录
dataset_dir = '/scratchdata1/users/a1978372/Anxuan/clam/CLAM/Processed_Data/dataset'
os.makedirs(dataset_dir, exist_ok=True)

# 进行KFold划分
for fold, (train_idx, val_idx) in enumerate(kf.split(range(dataset_size))):
    print(f"\nProcessing Fold {fold+1}...")

    # 获取训练集和验证集
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_features, train_labels = [], []
    for i in range(len(train_subset)):
        feature, label = train_subset[i]
        train_features.append(feature)
        train_labels.append(label)

    val_features, val_labels = [], []
    for i in range(len(val_subset)):
        feature, label = val_subset[i]
        val_features.append(feature)
        val_labels.append(label)

    # 确保 labels 是 Tensor 格式
    train_labels = [torch.tensor(lbl, dtype=torch.float32) for lbl in train_labels]
    val_labels = [torch.tensor(lbl, dtype=torch.float32) for lbl in val_labels]

    # 转换为 Tensor
    train_features_tensor = torch.stack(train_features)  # (num_samples, 5000, 256)
    train_labels_tensor = torch.stack(train_labels)  # (num_samples, label_dim)
    val_features_tensor = torch.stack(val_features)
    val_labels_tensor = torch.stack(val_labels)

    # 打印形状，确保正确
  

    # 保存数据
    torch.save(train_features_tensor, os.path.join(dataset_dir, f'fold_{fold+1}_train_features.pt'))
    torch.save(train_labels_tensor, os.path.join(dataset_dir, f'fold_{fold+1}_train_labels.pt'))
    torch.save(val_features_tensor, os.path.join(dataset_dir, f'fold_{fold+1}_val_features.pt'))
    torch.save(val_labels_tensor, os.path.join(dataset_dir, f'fold_{fold+1}_val_labels.pt'))


