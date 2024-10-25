import os
import sys
import torch
from tqdm import tqdm
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast

# 获取当前文件所在的绝对路径
current_path = os.path.abspath(__file__)

# 获取当前文件所在目录的父目录的绝对路径
parent_dir = os.path.dirname(os.path.dirname(current_path))

# 将父目录添加到sys.path中
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir,'model'))
sys.path.append(os.path.join(parent_dir,'utils_eval'))

from model.Integrated_module import Integrated_module # 导入Integrated_module模型
from utils_eval.utils import integrated_module_dataloader, LabelSmoothing

# 现在可以从A目录中导入config.py
from config import Config, parser
opt = Config()
args = parser.parse_args()  # 获取所有参数

backbone = args.backbone
log_dir = os.path.join(opt.log_dir,'Integrated_module_log',backbone) # 设置保存log的路径
os.makedirs(log_dir,exist_ok=True) # 创建该路径
Integrated_module_loss_writer = SummaryWriter(log_dir=log_dir) # 设置一个SummaryWriter便于在tensorboard上查看
MSRVTT_feature_path = opt.MSRVTT_features_path # 存储MSRVTT特征的h5文件路径
# MSRVTT_feature_path = args.MSRVTT_features_path # 存储MSRVTT特征的h5文件路径
MSVD_feature_path = opt.MSVD_features_path # 存储MSVD特征的h5文件路径
# MSVD_feature_path = args.MSVD_features_path # 存储MSVD特征的h5文件路径
captions_h5_path = opt.captions_path # 存储了处理后的captions的hdf5格式的文件
MSRVTT_model_weights_path = os.path.join(opt.model_weights_path,'Integrated_module',backbone,'MSRVTT') # 设置在MSRVTT训练后的权重保存路径
MSVD_model_weights_path = os.path.join(opt.model_weights_path,'Integrated_module',backbone,'MSVD') # 设置在MSVD训练后的权重保存路径
os.makedirs(MSRVTT_model_weights_path,exist_ok=True) # 创建该路径
os.makedirs(MSVD_model_weights_path,exist_ok=True) # 创建该路径

# 准备训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# Config
batch_size = opt.batch_size # 256
lr = opt.lr # 1e-4
d_model = args.d_model # 512
d_ff =  opt.d_ff # 2048
n_layers = opt.n_layers # 6
heads =  opt.heads # 8
dropout_rate = opt.dropout_rate # 0.1
MSRVTT_num_epochs = opt.MSRVTT_num_epochs # 25
MSVD_num_epochs = opt.MSVD_num_epochs # 25
PAD_ID =  opt.PAD_ID # 0

# 获取词表长度
vocab_length = opt.vocab_size

# 在MSRVTT数据集上的训练
MSRVTT_output_dim = vocab_length
model = Integrated_module(MSRVTT_output_dim,d_model,d_ff,n_layers,heads,PAD_ID,dropout_rate,device)

# 创建 GradScaler 实例
scaler_MSRVTT = GradScaler()

# 优化器和损失函数
criterion_MSRVTT = LabelSmoothing(size=MSRVTT_output_dim, padding_idx=PAD_ID, smoothing=0.1)
def loss_func(finall_pred,  v_pred, n_pred, tgt, v_tgt, n_tgt):
    # 将预测值展平为二维张量
    finall_pred_flat = finall_pred.contiguous().view(-1, finall_pred.size(-1))
    # 将目标序列展平为一维张量
    tgt_flat = tgt.contiguous().view(-1)
    # 计算损失
    base_loss = criterion_MSRVTT(finall_pred_flat, tgt_flat) / tgt.size(0)

    v_pred_flat = v_pred.contiguous().view(-1, v_pred.size(-1))
    v_tgt_flat = v_tgt.contiguous().view(-1)
    v_loss = criterion_MSRVTT(v_pred_flat, v_tgt_flat)  / v_tgt.size(0)

    n_pred_flat = n_pred.contiguous().view(-1, n_pred.size(-1))
    n_tgt_flat = n_tgt.contiguous().view(-1)
    n_loss = criterion_MSRVTT(n_pred_flat, n_tgt_flat)  / n_tgt.size(0)
    # print('===================================')
    # print(base_loss) # tensor(93.8767, device='cuda:0', grad_fn=<DivBackward0>)
    # print('===================================')
    # print(noun_loss) # tensor(38.0608, device='cuda:0', grad_fn=<DivBackward0>)
    # ss = input("press enter to continue")

    loss = base_loss  + n_loss +  v_loss
    return loss

optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in tqdm(range(MSRVTT_num_epochs)):
    model.train()  # 设置模型为训练模式
    torch.cuda.empty_cache() # 清理缓存，建议用在每个epoch开始时
    count = 0
    train_data_loss = 0
    for src, tgt,  v_q, v_tgt, n_q, n_tgt in integrated_module_dataloader(MSRVTT_feature_path, os.path.join(opt.captions_path,'MSRVTT.h5'), backbone, 'train', batch_size):
        src, tgt,  v_q, v_tgt, n_q, n_tgt = src.to(device), tgt.to(device), v_q.to(device), v_tgt.to(device), n_q.to(device), n_tgt.to(device) # 将数据移到设备上

        finall_input = tgt[:, :-1]
        v_input = v_q[:, :-1]
        n_input = n_q[:, :-1]
        
        optimizer.zero_grad()  # 清零梯度
        with autocast(device_type='cuda', dtype=torch.float16):
            outputs, verb_outs, noun_outs = model(src, finall_input, v_input, n_input)  # 前向传播
            finall_pred = model.generator(outputs)
            
            v_pred = model.generator(verb_outs)
            n_pred = model.generator(noun_outs)
            loss = loss_func(finall_pred,  v_pred,  n_pred, tgt[:, 1:],  v_tgt[:, 1:], n_tgt[:, 1:])  # 计算损失
        
        # 使用 scaler 对损失进行缩放
        scaler_MSRVTT.scale(loss).backward()

        # 更新权重
        scaler_MSRVTT.step(optimizer)
        # 更新 scaler 的状态
        scaler_MSRVTT.update()

        count += 1
        train_data_loss += loss

    train_data_loss = train_data_loss / count
    Integrated_module_loss_writer.add_scalar("MSRVTT_train_loss", train_data_loss, epoch)
    print(f"第{epoch + 1}轮的loss: {loss}")

    torch.save(model.state_dict(), f'{MSRVTT_model_weights_path}/model_epoch_{epoch + 1}.pth') # 保存权重文件

    model.eval()
    with torch.no_grad():
        valid_data_loss_ = 0
        n_count = 0
        for src, tgt,  v_q, v_tgt, n_q, n_tgt in integrated_module_dataloader(MSRVTT_feature_path, os.path.join(opt.captions_path,'MSRVTT.h5'), backbone, 'valid', batch_size):
            src, tgt,  v_q, v_tgt, n_q, n_tgt = src.to(device), tgt.to(device), v_q.to(device), v_tgt.to(device), n_q.to(device), n_tgt.to(device) # 将数据移到设备上
            
            finall_input = tgt[:, :-1]
            v_input = v_q[:, :-1]
            n_input = n_q[:, :-1]
            
            optimizer.zero_grad()  # 清零梯度
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs, verb_outs, noun_outs = model(src, finall_input, v_input, n_input)  # 前向传播
                finall_pred = model.generator(outputs)
                v_pred = model.generator(verb_outs)
                n_pred = model.generator(noun_outs)
                loss = loss_func(finall_pred,  v_pred,  n_pred, tgt[:, 1:],  v_tgt[:, 1:], n_tgt[:, 1:])  # 计算损失
            valid_data_loss_ += loss
            n_count += 1

        valid_data_loss_ = valid_data_loss_ / n_count
    Integrated_module_loss_writer.add_scalar("MSRVTT_valid_loss", valid_data_loss_, epoch)


# 在MSVD数据集上的训练
MSVD_output_dim = vocab_length
model = Integrated_module(MSVD_output_dim,d_model,d_ff,n_layers,heads,PAD_ID,dropout_rate,device)

# 创建 GradScaler 实例
scaler_MSVD = GradScaler()

# 优化器和损失函数
criterion_MSVD = LabelSmoothing(size=MSVD_output_dim, padding_idx=PAD_ID, smoothing=0.1)

def loss_func(finall_pred,  v_pred, n_pred, tgt, v_tgt, n_tgt):
    # 将预测值展平为二维张量
    finall_pred_flat = finall_pred.contiguous().view(-1, finall_pred.size(-1))
    # 将目标序列展平为一维张量
    tgt_flat = tgt.contiguous().view(-1)
    # 计算损失
    base_loss = criterion_MSVD(finall_pred_flat, tgt_flat) / tgt.size(0)

    v_pred_flat = v_pred.contiguous().view(-1, v_pred.size(-1))
    v_tgt_flat = v_tgt.contiguous().view(-1)
    v_loss = criterion_MSVD(v_pred_flat, v_tgt_flat)  / v_tgt.size(0)

    n_pred_flat = n_pred.contiguous().view(-1, n_pred.size(-1))
    n_tgt_flat = n_tgt.contiguous().view(-1)
    n_loss = criterion_MSVD(n_pred_flat, n_tgt_flat)  / n_tgt.size(0)

    loss = base_loss + n_loss +  v_loss
    return loss

optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in tqdm(range(MSVD_num_epochs)):
    model.train()  # 设置模型为训练模式
    torch.cuda.empty_cache() # 清理缓存，建议用在每个epoch开始时
    count = 0
    train_data_loss = 0
    for src, tgt,  v_q, v_tgt, n_q, n_tgt in integrated_module_dataloader(MSVD_feature_path, os.path.join(opt.captions_path,'MSVD.h5'), backbone, 'train', batch_size):
        src, tgt,  v_q, v_tgt, n_q, n_tgt = src.to(device), tgt.to(device), v_q.to(device), v_tgt.to(device), n_q.to(device), n_tgt.to(device) # 将数据移到设备上
        finall_input = tgt[:, :-1]
        v_input = v_q[:, :-1]
        n_input = n_q[:, :-1]
        
        optimizer.zero_grad()  # 清零梯度
        with autocast(device_type='cuda', dtype=torch.float16):
            outputs, verb_outs, noun_outs = model(src, finall_input, v_input, n_input)  # 前向传播
            finall_pred = model.generator(outputs)
            
            v_pred = model.generator(verb_outs)
            n_pred = model.generator(noun_outs)
            loss = loss_func(finall_pred,  v_pred,  n_pred, tgt[:, 1:],  v_tgt[:, 1:], n_tgt[:, 1:])  # 计算损失
    
        # 使用 scaler 对损失进行缩放
        scaler_MSVD.scale(loss).backward()

        # 更新权重
        scaler_MSVD.step(optimizer)
        # 更新 scaler 的状态
        scaler_MSVD.update()

        count += 1
        train_data_loss += loss

    train_data_loss = train_data_loss / count
    Integrated_module_loss_writer.add_scalar("MSVD_train_loss", train_data_loss, epoch)
    print(f"第{epoch + 1}轮的loss: {loss}")

    torch.save(model.state_dict(), f'{MSVD_model_weights_path}/model_epoch_{epoch + 1}.pth')

    model.eval()
    with torch.no_grad():
        valid_data_loss_ = 0
        n_count = 0
        for src, tgt,  v_q, v_tgt, n_q, n_tgt in integrated_module_dataloader(MSVD_feature_path, os.path.join(opt.captions_path,'MSVD.h5'), backbone, 'valid', batch_size):
            src, tgt,  v_q, v_tgt, n_q, n_tgt = src.to(device), tgt.to(device), v_q.to(device), v_tgt.to(device), n_q.to(device), n_tgt.to(device) # 将数据移到设备上
            finall_input = tgt[:, :-1]
            v_input = v_q[:, :-1]
            n_input = n_q[:, :-1]
            
            optimizer.zero_grad()  # 清零梯度
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs, verb_outs, noun_outs = model(src, finall_input, v_input, n_input)  # 前向传播
                finall_pred = model.generator(outputs)
                
                v_pred = model.generator(verb_outs)
                n_pred = model.generator(noun_outs)
                loss = loss_func(finall_pred,  v_pred, n_pred, tgt[:, 1:],  v_tgt[:, 1:], n_tgt[:, 1:])  # 计算损失
            valid_data_loss_ += loss
            n_count += 1

        valid_data_loss_ = valid_data_loss_ / n_count
    Integrated_module_loss_writer.add_scalar("MSVD_valid_loss", valid_data_loss_, epoch)