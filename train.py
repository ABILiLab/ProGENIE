import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from models import MLPModel
from evaluation import evaluate_model  # 导入evaluate_model函数

# 设置参数
epochs = 100
learning_rate = 1e-4
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 假设您已经设定了数据路径
dataset_dir = '/scratchdata1/users/a1978372/Anxuan/clam/CLAM/Processed_Data/dataset'
results_dir = '/scratchdata1/users/a1978372/Anxuan/clam/CLAM/hax/train/results'

# 创建结果保存的目录
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# 用于存储每个fold在80-100 epoch的Pearson相关系数
fold_pearson_corrs = []

# 五折交叉验证
for fold in range(1, 6):
    print(f"Processing fold {fold}...")
    
    # 加载训练和验证数据
    train_features = torch.load(os.path.join(dataset_dir, f'fold_{fold}_train_features.pt'))
    train_labels = torch.load(os.path.join(dataset_dir, f'fold_{fold}_train_labels.pt'))
    val_features = torch.load(os.path.join(dataset_dir, f'fold_{fold}_val_features.pt'))
    val_labels = torch.load(os.path.join(dataset_dir, f'fold_{fold}_val_labels.pt'))

    # 将数据转换为TensorDataset
    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)

    # 创建DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    input_dim = 500 * 256  # 输入的展平后的维度
    output_dim = 13591  # 输出的维度
    model = MLPModel(input_dim, output_dim).to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 记录训练和验证的损失及评估指标
    train_losses = []
    val_losses = []
    train_pearson_corrs = []
    val_pearson_corrs = []
    train_rmses = []
    val_rmses = []

    # 训练过程
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0

        # 训练阶段
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)  # 将输入和标签移动到正确的设备

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        # 计算训练集损失
        train_loss = running_train_loss / len(train_dataloader)
        train_losses.append(train_loss)

        # 在验证集上评估模型
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)  # 确保验证集数据在正确的设备上
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

        # 计算验证集损失
        val_loss = running_val_loss / len(val_dataloader)
        val_losses.append(val_loss)

        # 计算 Pearson Correlation 和 RMSE
        train_pearson_corr, train_rmse = evaluate_model(model, train_dataloader, device)
        val_pearson_corr, val_rmse = evaluate_model(model, val_dataloader, device)

        # 记录评估指标
        train_pearson_corrs.append(train_pearson_corr)
        val_pearson_corrs.append(val_pearson_corr)
        train_rmses.append(train_rmse)
        val_rmses.append(val_rmse)

        # 打印每个epoch的损失和评估指标
        print(f"Epoch {epoch+1}/{epochs}, Train_Loss: {train_loss:.4f}, "
              f"Val_Loss: {val_loss:.4f}, "
              f"Train_Pearson: {train_pearson_corr:.4f}, "
              f"Val_Pearson: {val_pearson_corr:.4f}, "
              f"Train_RMSE: {train_rmse:.4f}, "
              f"Val_RMSE: {val_rmse:.4f}")

    # 计算80-100 epoch的平均Pearson相关系数
    avg_pearson_corr = np.mean(val_pearson_corrs[79:100])
    fold_pearson_corrs.append(avg_pearson_corr)

    # 为当前fold创建结果文件夹
    fold_results_dir = os.path.join(results_dir, f"fold{fold}")
    os.makedirs(fold_results_dir, exist_ok=True)

    # 绘制训练和验证损失图
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), train_losses, label='Training Loss', color='blue')
    plt.plot(range(epochs), val_losses, label='Validation Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(fold_results_dir, 'loss_plot.jpg'))
    plt.close()

    # 绘制训练和验证 Pearson Correlation 图
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), train_pearson_corrs, label='Train Pearson Correlation', color='green')
    plt.plot(range(epochs), val_pearson_corrs, label='Validation Pearson Correlation', color='orange')
    plt.title('Training and Validation Pearson Correlation')
    plt.xlabel('Epochs')
    plt.ylabel('Pearson Correlation')
    plt.legend()
    plt.savefig(os.path.join(fold_results_dir, 'pearson_plot.jpg'))
    plt.close()

    # 绘制训练和验证 RMSE 图
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), train_rmses, label='Train RMSE', color='purple')
    plt.plot(range(epochs), val_rmses, label='Validation RMSE', color='brown')
    plt.title('Training and Validation RMSE')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig(os.path.join(fold_results_dir, 'rmse_plot.jpg'))
    plt.close()

    print(f"Results for fold {fold} saved to {fold_results_dir}")

# 绘制五折交叉验证的Pearson相关系数小提琴图
plt.figure(figsize=(10, 6))
sns.violinplot(data=fold_pearson_corrs, color='lightblue')
plt.title('Pearson Correlation (Epochs 80-100) Across Folds')
plt.ylabel('Pearson Correlation')
plt.xlabel('Folds')
plt.savefig(os.path.join(results_dir, 'pearson_violin_plot.jpg'))
plt.show()
