import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=8, num_layers=3, input_days=90, output_days=90):
        super(TransformerModel, self).__init__()
        
        # 输入特征的线性投影
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len= input_days)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 解码器（用于生成预测）
        self.decoder = nn.Sequential(
            nn.Linear(d_model * input_days, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_days)
        )
        
        
    def forward(self, src):
        # src shape: [batch_size, input_days, input_dim]
        src = src.permute(1, 0, 2)  # [input_days, batch_size, input_dim]
        
        # 投影到d_model维度
        src = self.input_proj(src)  # [input_days, batch_size, d_model]
        
        # 添加位置编码
        src = self.pos_encoder(src)  # [input_days, batch_size, d_model]
        
        # 通过Transformer编码器
        output = self.transformer_encoder(src)  # [input_days, batch_size, d_model]
        
        # 重塑输出以用于解码器
        output = output.permute(1, 0, 2).contiguous()  # [batch_size, input_days, d_model]
        output = output.view(output.size(0), -1)  # [batch_size, input_days * d_model]
        
        # 通过解码器生成预测
        output = self.decoder(output)  # [batch_size, onput_days]
        
            
        return output
    def compute_loss(self, prediction, target):
        # 计算预测损失
        pred_loss = nn.MSELoss()(prediction, target)
        return pred_loss

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)  # 形状: [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # 形状: [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度用正弦
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度用余弦
        
        # 关键修改：将pe的维度调整为 [max_len, 1, d_model]，支持广播到batch_size
        self.register_buffer('pe', pe.unsqueeze(1))  
        
    def forward(self, x):
        # x 形状: [input_days, batch_size, d_model]
        # 截取与x的input_days匹配的位置编码，形状: [input_days, 1, d_model]
        x = x + self.pe[:x.size(0), :, :]  # 修正切片方式，从第一维度取input_days
        return x

class ImprovedTransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=8, num_layers=3, input_days=90, output_days=90, latent_dim=64):
        super(ImprovedTransformerModel, self).__init__()
        self.input_days = input_days
        # 输入特征的线性投影
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=input_days)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # VAE组件
        self.vae_mu = nn.Sequential(
            nn.Linear(input_dim * input_days, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, latent_dim)
        )
        self.vae_logvar = nn.Sequential(
            nn.Linear(input_dim * input_days, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, latent_dim)
        )
        self.vae_decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, input_dim * input_days)
        )
        
        # 解码器1（用于生成初步预测）
        self.prediction = nn.Sequential(
            nn.Linear(d_model * input_days, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_days)
        )

        # 解码器2（用于生成预测的残差）
        self.residual_prediction = nn.Sequential(
            nn.Linear(d_model * input_days, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_days)
        )
        
   
        
        self.latent_dim = latent_dim
    

    def transformer_encode(self, src):
        # src shape: [batch_size, input_days, input_dim]
        src = src.permute(1, 0, 2) 
        
        # 投影到d_model维度
        src = self.input_proj(src)  # [input_days, batch_size, d_model]
        
        # 添加位置编码
        src = self.pos_encoder(src)  # [input_days, batch_size, d_model]
        
        # 通过Transformer编码器
        output = self.transformer_encoder(src)  # [input_days, batch_size, d_model]
        
        # 重塑输出
        output = output.permute(1, 0, 2).contiguous()  # [batch_size, input_days, d_model]

        return output

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    

    def vae_encode(self, src):
        src = src.view(src.size(0), -1)  # [batch_size, input_days * input_dim]
        # 计算VAE的均值和方差
        mu = self.vae_mu(src)
        logvar = self.vae_logvar(src)
        return mu, logvar
        
    def vae_decode(self, src):
        output = self.vae_decoder(src)
        output = output.view(src.size(0), self.input_days, -1)
        return output
        

        
    def forward(self, src):
        transformer_output = self.transformer_encode(src)
        mu, logvar = self.vae_encode(src)
        z = self.reparameterize(mu, logvar)
        reconstruction_src = self.vae_decode(z)
        reconstruction_transformer_output = self.transformer_encode(reconstruction_src)
        transformer_output = transformer_output.view(src.size(0), -1)
        prediction = self.prediction(transformer_output)
        reconstruction_transformer_output = reconstruction_transformer_output.view(src.size(0), -1)
        residual = self.residual_prediction(reconstruction_transformer_output)
        prediction = prediction + residual
        return prediction, reconstruction_src, mu, logvar
        
    def compute_loss(self, prediction, target, reconstruction, src, mu, logvar, recon_weight=0.01, kl_weight=0.001):
        # 计算预测损失
        pred_loss = nn.MSELoss()(prediction, target)
        
        # 计算重构损失
        recon_loss = nn.MSELoss()(reconstruction, src)
        
        # 计算KL散度
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_div = kl_div / prediction.size(0)  # 归一化到批次大小
        
        # 总损失
        loss = pred_loss + recon_weight * recon_loss + kl_weight * kl_div
        
        return loss
