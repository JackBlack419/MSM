import os
import sys
import copy
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

# 特征融合
class FeatureFusion(nn.Module):
    def __init__(self, input_dim):
        """
        初始化特征融合模块。
        
        参数:
        - input_dim (int): 单个输入特征的维度
        """
        super(FeatureFusion, self).__init__()
        self.linear = nn.Linear(2 * input_dim, input_dim)  # 线性层用于将拼接后的特征映射到原始维度
        self.relu = nn.ReLU()
    def forward(self, x1, x2):
        """
        前向传播。
        
        参数:
        - x1, x2, x3 (Tensor): 输入的三个特征张量，形状为(batch_size, 20, input_dim)
        
        返回:
        - Tensor: 融合后的特征张量，形状为(batch_size, 20, input_dim)
        """
        # 拼接三个输入特征张量
        x_concat = torch.cat([x1, x2], dim=-1)  # 拼接在最后一维
        
        # 通过线性层进行特征融合
        x_fused = self.relu(self.linear(x_concat))
        
        return x_fused

# 定义一个类，用于生成输出的log_softmax
class Generator(nn.Module):
    def __init__(self,d_model,output_dim,device):
        super(Generator, self).__init__()
        self.fc = nn.Linear(d_model, output_dim,device=device)

    def forward(self, x):
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# 定义一个基线模型类，用于组合上述定义的各个模块
class With_verb_block(nn.Module):
    def __init__(self, vocab_size, d_model, d_feedforward, num_layers, num_heads, pad_idx, dropout_prob, device):
        super(With_verb_block,self).__init__()

        self.pad_idx = pad_idx
        self.caption_embedding = Caption_Embedding(d_model,vocab_size,device)
        self.position_embedding = PositionalEncoding(d_model,dropout_prob,device)

        # self.feature_fusion = FeatureFusion(input_dim=d_model).to(device)
        layer_norm = nn.LayerNorm(d_model).to(device)
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, d_feedforward, dropout_prob,batch_first=True).to(device)
        decoder_layer = nn.TransformerDecoderLayer(d_model, num_heads, d_feedforward, dropout_prob,batch_first=True).to(device)
        self.encoders = _get_clones(nn.TransformerEncoder(encoder_layer, num_layers, layer_norm),2)
        self.encoder = self.encoders[0].to(device)
        self.verb_encoder = self.encoders[1].to(device)
        self.decoders = _get_clones(nn.TransformerDecoder(decoder_layer, 2, layer_norm),2)
        self.decoder = self.decoders[0].to(device)
        self.verb_decoder = self.decoders[1].to(device)
        self.generator = Generator(d_model, vocab_size,device)
        self._initialize_weights()
    # 生成一个mask矩阵，用于遮蔽未来位置的信息
    def generate_square_subsequent_mask(self,sz,device):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(device)

    # 编码函数
    def base_encode(self,src):
        src = self.encoder(src)
        return src
    
    def verb_encode(self,src):
        verb_src = self.verb_encoder(src)
        return verb_src
    
    def encode(self,src):
        src = self.base_encode(src) + self.verb_encode(src)
        return src

    # 解码函数，包含了解码器的整个操作流程
    def decode(self,src, tgt):
        # 生成一个遮蔽mask，用于屏蔽padding位置的信息
        tgt_pad_mask = tgt == self.pad_idx
        tgt_pad_mask = tgt_pad_mask.to(tgt.device)
        # 生成一个遮蔽mask，用于屏蔽未来位置的信息
        tgt_mask = self.generate_square_subsequent_mask(tgt.shape[1], device=tgt.device)

        # 将token序列转换为embedding表示，并加上位置编码
        tgt_embeded = self.position_embedding(self.caption_embedding(tgt))

        
        # 调用解码器进行解码
        out = self.decoder(tgt_embeded, src, tgt_mask, None, tgt_pad_mask, None)
        return out
    
    def verb_decode(self,verb_src,verb_tgt):

        # 生成一个遮蔽mask，用于屏蔽padding位置的信息
        verb_tgt_pad_mask = verb_tgt == self.pad_idx
        verb_tgt_pad_mask = verb_tgt_pad_mask.to(verb_tgt.device)
        # 生成一个遮蔽mask，用于屏蔽未来位置的信息
        verb_tgt_mask = self.generate_square_subsequent_mask(verb_tgt.shape[1], device=verb_tgt.device)
        
        # 将token序列转换为embedding表示，并加上位置编码
        verb_tgt_embeded = self.position_embedding(self.caption_embedding(verb_tgt))
        verb_out = self.verb_decoder(verb_tgt_embeded, verb_src, verb_tgt_mask, None, verb_tgt_pad_mask, None)

        return verb_out
    # 整体前向传播流程
    def forward(self, src, tgt, verb_tgt):
        verb_src = self.verb_encode(src)
        verb_out = self.verb_decode(verb_src,verb_tgt)
        finall_src = self.encode(src)
        out = self.decode(finall_src,tgt)
        return out, verb_out
    
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