import torch

from torch.utils.data import Dataset
# 创建时间序列数据集


class TimeSeriesDataset(Dataset):
    def __init__(self, data, input_days=90, output_days=90, target_col='Global_active_power'):
        self.data = data
        self.input_days = input_days
        self.output_days = output_days
        self.target_col = target_col
        self.target_idx = data.columns.get_loc(target_col)
        self.length = len(data) - input_days - output_days + 1
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # 输入序列（特征和目标）
        input_seq = self.data.iloc[idx:idx+self.input_days].values
        # 目标序列（仅目标变量）
        target_seq = self.data.iloc[idx+self.input_days:idx+self.input_days+self.output_days, self.target_idx].values
        
        return {
            'input': torch.FloatTensor(input_seq),
            'target': torch.FloatTensor(target_seq)
        }