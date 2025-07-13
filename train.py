import numpy as np
import torch
from evaluate import evaluate_model
import argparse
from utils import convert_minutes_to_days, draw_train_val_loss, get_gpu_memory_usage, integrate_results, draw_predictions_vs_ground_truth, count_parameters
import pandas as pd
from time_datasets import TimeSeriesDataset
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from model.transformer import TransformerModel, ImprovedTransformerModel
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os   
from tqdm import tqdm
import time


# 训练函数
def train_model(model, train_loader, val_loader, optimizer, device, epochs, scaler, target_col, save_dir, model_name, use_improved):
    best_val_loss = float('inf')
    mae_results = []
    mse_results = []
    train_losses = []
    val_losses = []
    
    best_model = None
    best_epoch = 0
    gpu_memory_usage = []
    # 创建总进度条
    epoch_progress = tqdm(range(epochs), desc=f'Training {model_name}', unit='epoch')
    
    for epoch in epoch_progress:
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:

            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            if use_improved:
                predictions, reconstructions, mu, logvar = model(inputs)
            else:
                predictions = model(inputs)

            if use_improved:
                loss = model.compute_loss(predictions, targets, reconstructions, inputs, mu, logvar)
            else:
                loss = model.compute_loss(predictions, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)
                
                if use_improved:
                    predictions, reconstructions, mu, logvar = model(inputs)
                else:
                    predictions = model(inputs)

                if use_improved:
                    loss = model.compute_loss(predictions, targets, reconstructions, inputs, mu, logvar)
                else:
                    loss = model.compute_loss(predictions, targets)
                
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss) 
        
        # 评估模型
        evaluation = evaluate_model(
            model=model,
            test_loader=val_loader,
            scaler=scaler,
            device=device,
            target_col=target_col,
            use_improved=use_improved
        )
        mse_results.append(evaluation["mse"])
        mae_results.append(evaluation["mae"])
        
        # 更新总进度条信息
        epoch_progress.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'val_mae': f'{evaluation["mae"]:.4f}',
            'val_mse': f'{evaluation["mse"]:.4f}'
        })
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            best_epoch = epoch + 1
        
        

        
        memory_usage = get_gpu_memory_usage()
        gpu_memory_usage.append(memory_usage)
            
        
            
    
    # 关闭总进度条
    epoch_progress.close()
    
    # 绘制训练和验证损失曲线
    draw_train_val_loss(train_losses, val_losses, save_dir, model_name)
    
    # 计算平均指标
    avg_mae = np.mean(mae_results)
    std_mae = np.std(mae_results)
    avg_mse = np.mean(mse_results)
    std_mse = np.std(mse_results)

    print(f'{model_name[0].upper() + model_name[1:]} Model, best validation loss: {best_val_loss:.4f}, best epoch: {best_epoch}')
    print(f'{model_name[0].upper() + model_name[1:]} Model, Average MAE: {avg_mae:.4f}, Std MAE: {std_mae:.4f}')
    print(f'{model_name[0].upper() + model_name[1:]} Model, Average MSE: {avg_mse:.4f}, Std MSE: {std_mse:.4f}')
    print(f'{model_name[0].upper() + model_name[1:]} Model, GPU Memory Usage: {np.mean(gpu_memory_usage)} MB')
    # 保存最佳模型
    save_path = f"{save_dir}/{model_name}_best_model.pth"
    torch.save(best_model, save_path)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, default="data/train.csv")
    parser.add_argument('--test_data_path', type=str, default="data/test.csv")
    parser.add_argument('--save_dir', type=str, default="data/output")
    parser.add_argument('--model_name', type=str, default="transformer")
    parser.add_argument('--use_improved', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--input_days', type=int, default=90)
    parser.add_argument('--output_days', type=int, default=90)
    parser.add_argument('--target_col', type=str, default='Global_active_power')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)
    np.random.seed(42)

    train_df = convert_minutes_to_days(args.train_data_path)
    test_df = convert_minutes_to_days(args.test_data_path)
    
    # 合并训练集和测试集进行标准化
    all_data = pd.concat([train_df, test_df], ignore_index=True)
    
    # 提取特征列（除日期列外的所有列）
    feature_cols = all_data.columns.drop('date')
    
    # 标准化数据
    scaler = StandardScaler()
    all_data_scaled = pd.DataFrame(scaler.fit_transform(all_data[feature_cols]), columns=feature_cols)
    all_data_scaled['date'] = all_data['date']
    
    # 分离回训练集和测试集
    train_size = len(train_df)
    train_data_scaled = all_data_scaled.iloc[:train_size]
    test_data_scaled = all_data_scaled.iloc[train_size:]
    
    
    train_dataset = TimeSeriesDataset(
        train_data_scaled.drop('date', axis=1), 
        input_days=args.input_days, 
        output_days=args.output_days,
        target_col=args.target_col
    )
    
    test_dataset = TimeSeriesDataset(
        test_data_scaled.drop('date', axis=1), 
        input_days=args.input_days, 
        output_days=args.output_days,
        target_col=args.target_col
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型
    input_dim = len(feature_cols)
    
    if args.use_improved:
        model = ImprovedTransformerModel(
            input_dim=input_dim,
            d_model=64,
            nhead=8,
            num_layers=3,
            input_days=args.input_days,
            output_days=args.output_days
        ).to(device)
    else:
        model = TransformerModel(
            input_dim=input_dim,
            d_model=64,
            nhead=8,
            num_layers=3,
            input_days=args.input_days,
            output_days=args.output_days
        ).to(device)


    summary(model, input_size=(90, 13), device=device.type)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
     # 记录训练开始时间
    start_time = time.time()
    print(f"{args.model_name[0].upper() + args.model_name[1:]} Model 训练开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print(f"{args.model_name[0].upper() + args.model_name[1:]} Model 模型参数总数: {count_parameters(model)}")

    # 训练模型
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,  
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        scaler=scaler,
        target_col=args.target_col,
        save_dir=args.save_dir,
        model_name=args.model_name,
        use_improved=args.use_improved
    )

    # 记录训练结束时间
    end_time = time.time()
    print(f"{args.model_name[0].upper() + args.model_name[1:]} Model 训练结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    
    # 计算并打印训练总耗时
    training_time = end_time - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"{args.model_name[0].upper() + args.model_name[1:]} Model 训练总耗时: {int(hours):02d}小时 {int(minutes):02d}分钟 {seconds:.2f}秒")
    

    
    
    # 评估模型
    evaluation = evaluate_model(
        model=model,
        test_loader=test_loader,
        scaler=scaler,
        device=device,
        target_col=args.target_col,
        use_improved=args.use_improved
    )
    predictions = evaluation['predictions']
    targets = evaluation['targets']
    
    final_preds, final_targets = integrate_results(predictions, targets)

    draw_predictions_vs_ground_truth(final_preds, final_targets, args.save_dir, args.model_name)

    print(f'{args.model_name[0].upper() + args.model_name[1:]} Model, Best MAE: {mean_absolute_error(final_targets, final_preds):.4f}')
    print(f'{args.model_name[0].upper() + args.model_name[1:]} Model, Best MSE: {mean_squared_error(final_targets, final_preds):.4f}')

    
