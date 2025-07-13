import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
import subprocess
import os


def convert_minutes_to_days(filepath):
    df = pd.read_csv(
        filepath,
        parse_dates=["DateTime"],  # 解析DateTime列为时间格式
        infer_datetime_format=True,  # 自动识别时间格式（如2008/12/13 21:38:00）
        na_values=['?']
    )

    columns_to_convert = df.columns.drop('DateTime')
                
                
    df[columns_to_convert] = df[columns_to_convert].astype('float64')


    df = df.ffill().bfill()



    # 计算剩余分表能耗（sub_metering_remainder）
    df["Sub_metering_remainder"] = (df["Global_active_power"] * 1000 / 60) - (
        df["Sub_metering_1"] + df["Sub_metering_2"] + df["Sub_metering_3"]
    )


    # 从DateTime中提取日期（精确到天）
    df["date"] = df["DateTime"].dt.date  # 格式：YYYY-MM-DD（日期对象）



    # 按日期分组，执行指定的聚合操作
    daily_df = df.groupby("date").agg(
        # 总和类变量（按天取总和）
        Global_active_power=("Global_active_power", "sum"),
        Global_reactive_power=("Global_reactive_power", "sum"),
        Sub_metering_1=("Sub_metering_1", "sum"),
        Sub_metering_2=("Sub_metering_2", "sum"),
        Sub_metering_3=("Sub_metering_3", "sum"),
        Sub_metering_remainder=("Sub_metering_remainder", "sum"),
        # 平均值类变量（按天取平均）
        Voltage=("Voltage", "mean"),
        Global_intensity=("Global_intensity", "mean"),
        # 气象类变量（取当天任意一个值，用first即可）
        RR=("RR", "first"),
        NBJRR1=("NBJRR1", "first"),
        NBJRR5=("NBJRR5", "first"),
        NBJRR10=("NBJRR10", "first"),
        NBJBROU=("NBJBROU", "first")
    ).reset_index()

    # 将date列转换为datetime格式（便于后续时间特征提取）
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df["RR"] = daily_df["RR"] / 10

    return daily_df

def draw_train_val_loss(train_losses, val_losses, save_dir, model_name):

    epochs = len(train_losses)
    
    plt.figure(figsize=(12, 6), dpi=100)
    
    # 绘制训练损失
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', color='blue', linewidth=2)
    
    # 绘制验证损失
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss', color='orange', linewidth=2)
    
    # 设置标题和标签
    plt.title(f'{model_name[0].upper() + model_name[1:]} Model, Training and Validation Loss Over Epochs', fontsize=16, pad=20)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    
    # 添加网格
    plt.grid(alpha=0.3)
    
    # 添加图例
    plt.legend(fontsize=12)
    
    # 紧凑布局
    plt.tight_layout()
    
    save_path = f"{save_dir}/{model_name}_train_val_loss.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')







def get_gpu_memory_usage():
    try:
        # 获取当前 Python 进程 ID
        current_pid = os.getpid()
        
        # 执行 nvidia-smi 命令
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,nounits,noheader"],
            capture_output=True,
            text=True
        )
        
        # 解析输出
        lines = result.stdout.strip().split('\n')
        for line in lines:
            pid, memory = line.strip().split(',')
            if int(pid) == current_pid:
                return int(memory)  # 返回显存使用量（MB）
        
        return 0  # 未找到当前进程的显存信息
    except Exception as e:
        print(f"Error: {e}")
        return -1


def integrate_results(predictions, targets):

    window_num = predictions.shape[0]  
    window_size = predictions.shape[1]  
    total_days = window_size + window_num - 1 # 目标总天数605
    
    # 1. 计算最终预测结果（取重复预测的平均值）
    sum_pred = np.zeros(total_days, dtype=np.float64)  # 累加预测值
    count_pred = np.zeros(total_days, dtype=np.int32)  # 记录每个日期被预测的次数
    
    for i in range(window_num):  
        for k in range(window_size): 
            global_day = i + k  
            sum_pred[global_day] += predictions[i, k]
            count_pred[global_day] += 1
    
    # 计算平均值（确保无除零，因所有日期都被覆盖）
    final_predictions = sum_pred / count_pred
    
    # 2. 提取最终真实结果（每个日期的真实值唯一）
    final_targets = np.zeros(total_days, dtype=np.float64)
      # 遍历所有窗口和时间步，收集每个全局日期的真实值
    for i in range(window_num):
        for k in range(window_size):
            global_day = i + k
            final_targets[global_day] = targets[i, k]
    
    return final_predictions, final_targets


def draw_predictions_vs_ground_truth(final_preds, final_targets, save_dir, model_name):
    days = len(final_targets)
    start_date = datetime(2009, 4, 1)
    dates = [start_date + timedelta(days=i) for i in range(days)]
    
    # 创建图表
    plt.figure(figsize=(14, 7), dpi=100)

    # 绘制真实值曲线（蓝色实线）
    plt.plot(dates, final_targets, 
            color='#1f77b4', 
            linewidth=2.5, 
            label='Ground Truth')

    # 绘制预测值曲线（橙色虚线）
    plt.plot(dates, final_preds, 
            color='#ff7f0e', 
            linestyle='--', 
            linewidth=2,
            alpha=0.9,
            label='Prediction')

    # 设置标题和标签
    plt.title(f'{model_name[0].upper() + model_name[1:]} Model, Global_active_power: Prediction vs Ground Truth', fontsize=16, pad=20)
    plt.xlabel('Date (day)', fontsize=12)
    plt.ylabel('Global_active_power (kW)', fontsize=12)
    plt.grid(alpha=0.3)

    # 配置X轴日期格式
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))  # 每两个月一个刻度
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # 日期格式
    plt.xticks(rotation=30, ha='right')  # 旋转30度对齐

    # 添加图例
    plt.legend(fontsize=12)

    # 添加边距使图表更美观
    plt.margins(x=0.02)

    # 紧凑布局
    plt.tight_layout()

    save_path = f"{save_dir}/{model_name}_predictions_vs_ground_truth.png"

    plt.savefig(save_path, dpi=300, bbox_inches='tight')



def count_parameters(model):
    """计算模型的总参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)