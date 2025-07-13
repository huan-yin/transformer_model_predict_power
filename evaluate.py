import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error



# 评估模型
def evaluate_model(model, test_loader, scaler, device, target_col='Global_active_power', use_improved=True):
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input'].to(device)
            targets = batch['target'].cpu().numpy()
            if use_improved:
                predictions, _, _, _ = model(inputs)
            else:
                predictions = model(inputs)
            outputs = predictions.cpu().numpy()
            
            all_predictions.extend(outputs)
            all_targets.extend(targets)
    
    # 将预测和目标转换为numpy数组
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # 创建用于反标准化的虚拟数组
    dummy_predictions = np.zeros((all_predictions.size, test_loader.dataset.data.shape[1]))
    dummy_targets = np.zeros((all_targets.size, test_loader.dataset.data.shape[1]))
    
    target_idx = test_loader.dataset.data.columns.get_loc(target_col)
    
    # 重塑预测和目标以适应虚拟数组
    dummy_predictions[:, target_idx] = all_predictions.flatten()
    dummy_targets[:, target_idx] = all_targets.flatten()
    
    # 反标准化
    unscaled_predictions = scaler.inverse_transform(dummy_predictions)[:, target_idx]
    unscaled_targets = scaler.inverse_transform(dummy_targets)[:, target_idx]
    
    # 重塑回原始形状
    unscaled_predictions = unscaled_predictions.reshape(all_predictions.shape)
    unscaled_targets = unscaled_targets.reshape(all_targets.shape)
    
    # 计算评估指标
    mse = mean_squared_error(unscaled_targets.flatten(), unscaled_predictions.flatten())
    mae = mean_absolute_error(unscaled_targets.flatten(), unscaled_predictions.flatten())
    
    return {
        'mse': mse,
        'mae': mae,
        'predictions': unscaled_predictions,
        'targets': unscaled_targets
    }







