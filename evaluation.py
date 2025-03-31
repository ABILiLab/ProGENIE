import torch
import numpy as np
from sklearn.metrics import mean_squared_error

# 计算 Pearson's Correlation
def pearson_correlation(y_true, y_pred):
    """
    计算 Pearson’s correlation coefficient
    :param y_true: 真实标签
    :param y_pred: 预测值
    :return: Pearson's correlation coefficient
    """
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    
    # 计算 Pearson's correlation coefficient
    correlation = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
    return correlation

# 计算 Root Mean Squared Error (RMSE)
def rmse(y_true, y_pred):
    """
    计算 RMSE
    :param y_true: 真实标签
    :param y_pred: 预测值
    :return: RMSE
    """
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    
    # 计算 RMSE
    rmse_value = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse_value

# 评估函数
# 评估函数
def evaluate_model(model, dataloader, device):
    """
    在给定的dataloader上评估模型
    :param model: 训练好的模型
    :param dataloader: 数据加载器
    :param device: 使用的设备
    :return: Pearson’s Correlation, RMSE
    """
    model.eval()  # 切换到评估模式
    all_true = []
    all_pred = []
    
    with torch.no_grad():  # 不计算梯度
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)  # 确保数据在正确的设备上
            outputs = model(inputs)  # 得到模型的输出
            all_true.append(labels)
            all_pred.append(outputs)
    
    # 拼接所有的标签和预测值
    all_true = torch.cat(all_true, dim=0)  # 形状: (总样本数, 13591)
    all_pred = torch.cat(all_pred, dim=0)  # 形状: (总样本数, 13591)
    
    # 打印一些样本的形状以供检查
    print(f"Shape of all_true: {all_true.shape}")
    print(f"Shape of all_pred: {all_pred.shape}")

    # 计算 Pearson’s Correlation 和 RMSE
    pearson_corr = pearson_correlation(all_true, all_pred)
    rmse_value = rmse(all_true, all_pred)
    
    return pearson_corr, rmse_value
