import os
import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans

def perform_mini_batch_kmeans(features, n_clusters, batch_size=100):
    """对特征矩阵进行Mini-batch K-means聚类"""
    n_samples = features.shape[0]
    # 如果样本数小于聚类数，设置聚类数为样本数
    if n_samples < n_clusters:
        n_clusters = n_samples
    
    min_batch_kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=42)
    min_batch_kmeans.fit(features.numpy())  # 转换为numpy数组并进行聚类
    return min_batch_kmeans.cluster_centers_  # 返回簇中心

def save_centroids_to_file(centroids, save_path):
    """将聚类中心保存到文件"""
    torch.save(centroids, save_path)

# 文件夹路径
input_folder_path = '/scratchdata1/users/a1978372/Anxuan/clam/CLAM/Processed_Data/features'
output_folder_path = '/scratchdata1/users/a1978372/Anxuan/clam/CLAM/Processed_Data/features2'
# 目标聚类数
n_clusters = 500
# Mini-batch大小
batch_size = 100  # 每次批处理的大小，可以根据内存调整

# 如果输出文件夹不存在，创建它
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# 获取文件列表
file_list = os.listdir(input_folder_path)
file_paths = [os.path.join(input_folder_path, f) for f in file_list]

# 逐个处理每个文件
for file_path in file_paths:
    # 加载特征数据
    features = torch.load(file_path)

    # 使用Mini-batch K-means进行聚类，n_clusters是你希望得到的聚类中心数量
    centroids = perform_mini_batch_kmeans(features, n_clusters, batch_size)

    # 保存处理后的数据
    output_file_path = os.path.join(output_folder_path, os.path.basename(file_path))
    save_centroids_to_file(centroids, output_file_path)

print("处理完成，所有文件已保存到", output_folder_path)
