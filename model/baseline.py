import os
import sys
import math
# import clip
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

# 获取当前文件所在的绝对路径
current_path = os.path.abspath(__file__)

# 获取当前文件所在目录的父目录的绝对路径
parent_dir = os.path.dirname(os.path.dirname(current_path))

# 将父目录添加到sys.path中
sys.path.append(parent_dir)

# 现在可以从A目录中导入config.py
from config import Config
opt = Config()

CLIP_weights_path = opt.CLIP_weights_path # 存放了CLIPweights的路径


# 定义一个类，用于将输入的token序列转换为embedding表示
class Caption_Embedding(nn.Module):
    def __init__(self, d_model, vocab_size, device):
        super(Caption_Embedding, self).__init__()
        self.input_embedding = nn.Embedding(vocab_size, d_model, device=device)
        self.d_model = d_model

    def forward(self, x):
        return self.input_embedding(x) * math.sqrt(self.d_model)


# 定义一个类，用于生成位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, device, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0., max_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2, device=device) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# 定义一个类，用于生成输出的log_softmax
class Generator(nn.Module):
    def __init__(self,d_model,output_dim,device):
        super(Generator, self).__init__()
        self.fc = nn.Linear(d_model, output_dim,device=device)

    def forward(self, x):
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)


# 定义一个基线模型类，用于组合上述定义的各个模块
class BaseLine(nn.Module):
    def __init__(self, vocab_size, d_model, d_feedforward, num_layers, num_heads, pad_idx, dropout_prob, device):
        super(BaseLine,self).__init__()
        
        self.pad_idx = pad_idx
        self.caption_embedding = Caption_Embedding(d_model,vocab_size,device)
        self.position_embedding = PositionalEncoding(d_model,dropout_prob,device)

        layer_norm = nn.LayerNorm(d_model).to(device)
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, d_feedforward, dropout_prob,batch_first=True).to(device)
        decoder_layer = nn.TransformerDecoderLayer(d_model, num_heads, d_feedforward, dropout_prob,batch_first=True).to(device)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, layer_norm).to(device)
        self.decoder = nn.TransformerDecoder(decoder_layer, 2, layer_norm).to(device)
        self.generator = Generator(d_model, vocab_size,device)
        self._initialize_weights()

    # 生成一个mask矩阵，用于遮蔽未来位置的信息
    def generate_square_subsequent_mask(self,sz,device):
        mask = (torch.triu(torch.ones(sz, sz,)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(device)
    
    # 编码函数
    def encode(self,src):
        return self.encoder(self.position_embedding(src))

    # 解码函数，包含了解码器的整个操作流程
    def decode(self,src,tgt):
        # 生成一个遮蔽mask，用于屏蔽padding位置的信息
        tgt_pad_mask = tgt == self.pad_idx
        tgt_pad_mask = tgt_pad_mask.to(tgt.device)
        # 生成一个遮蔽mask，用于屏蔽未来位置的信息
        tgt_mask = self.generate_square_subsequent_mask(tgt.shape[1], device=tgt.device)

        # 将token序列转换为embedding表示，并加上位置编码
        tgt_embeded = self.position_embedding(self.caption_embedding(tgt))

        # 调用解码器进行解码
        out = self.decoder(tgt_embeded,src,tgt_mask,None,tgt_pad_mask,None)
        return out
    
    # 整体前向传播流程
    def forward(self,src,tgt):
        return self.decode(self.encode(src),tgt)
    
        # 初始化模型的所有权重
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.TransformerEncoderLayer) or isinstance(m, nn.TransformerDecoderLayer):
                for param in m.parameters():
                    if param.dim() > 1:
                        init.xavier_uniform_(param)