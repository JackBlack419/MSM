from nltk import word_tokenize, pos_tag

def extract_verbs(sentence):
    """从给定句子中提取动词"""
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    return {word for word, tag in tagged if tag.startswith('VB')}
    # return {word for word, tag in tagged if tag.startswith('NN')}

def calculate_average_inclusion_rate(sentences_A, sentences_B):
    inclusion_rates = []
    
    for i, sentence in enumerate(sentences_A):
        # 提取A中句子的动词
        verbs_A = extract_verbs(sentence)
        
        # 提取B中对应位置子列表的所有动词
        verbs_B = set()
        for sub_sentence in sentences_B[i]:
            verbs_B.update(extract_verbs(sub_sentence))
        
        # 计算包含率
        included_count = sum(1 for verb in verbs_A if verb in verbs_B)
        inclusion_rate = included_count / len(verbs_A) if verbs_A else 0
        inclusion_rates.append(inclusion_rate)
    
    # 计算平均包含率
    average_inclusion_rate = sum(inclusion_rates) / len(inclusion_rates) if inclusion_rates else 0
    return average_inclusion_rate

# # 示例数据
# sentences_A = ["I enjoy programming", "She enjoys to read books"]
# sentences_B = [["Programming is fun", "I enjoy coding"], ["Reading is a good habit", "She enjoys novels"]]

# 调用函数
# average_rate = calculate_average_inclusion_rate(sentences_A, sentences_B)
# print(f"Average Inclusion Rate: {average_rate}")

import os
import sys
import h5py
import json
import torch
from tqdm import tqdm
from transformers import BertTokenizer

# 获取当前文件所在的绝对路径
current_path = os.path.abspath(__file__)

# 获取当前文件所在目录的父目录的绝对路径
parent_dir = os.path.dirname(os.path.dirname(current_path))

# 将父目录添加到sys.path中
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir,'model'))
sys.path.append(os.path.join(parent_dir,'utils_eval'))

from model.Integrated_module import Integrated_module # 导入Integrated_module模型
from utils_eval.beam_search_code import beam_search # 导入beam search
from utils_eval.utils import load_test_src_tgt, language_eval

# 现在可以从A目录中导入config.py
from config import Config, parser
opt = Config()
args = parser.parse_args()  # 获取所有参数

backbone = args.backbone
tokenizer = BertTokenizer.from_pretrained(opt.BertTokenizer_path,max_len=opt.max_len,truncation=True,return_tensors='pt') # 使用bert的tokenizer
MSRVTT_feature_path = opt.MSRVTT_features_path # 存储MSRVTT特征的h5文件路径
# MSRVTT_feature_path = args.MSRVTT_features_path # 存储MSRVTT特征的h5文件路径
MSVD_feature_path = opt.MSVD_features_path # 存储MSVD特征的h5文件路径
# MSVD_feature_path = args.MSVD_features_path # 存储MSVD特征的h5文件路径
captions_h5_path = opt.captions_path # 存储了处理后的captions的hdf5格式的文件
MSRVTT_model_weights_path = os.path.join(opt.model_weights_path,'Integrated_module',backbone,'MSRVTT') # 设置在MSRVTT训练后的权重保存路径
MSVD_model_weights_path = os.path.join(opt.model_weights_path,'Integrated_module',backbone,'MSVD') # 设置在MSVD训练后的权重保存路径
MSRVTT_res_path = os.path.join(opt.test_and_inference_res,'Integrated_module',backbone,'MSRVTT') # 存放MSRVTT的inference result的路径
MSVD_res_path = os.path.join(opt.test_and_inference_res,'Integrated_module',backbone,'MSVD') # 存放MSVD的inference result的路径
os.makedirs(MSRVTT_res_path,exist_ok=True)
os.makedirs(MSVD_res_path,exist_ok=True)


# 准备测试
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Config
batch_size = 256
d_model = args.d_model # 512
d_ff = 2048
n_layers = 6
heads = 8
dropout_rate = 0.1
PAD_ID = 0
BOS_ID = 101
EOS_ID = 102


# 获取词表长度
vocab_length = opt.vocab_size

# 在MSVD数据集上的测试
MSVD_output_dim = vocab_length
model = Integrated_module(MSVD_output_dim,d_model,d_ff,n_layers,heads,PAD_ID,dropout_rate,device)

epoch = 8
model.load_state_dict(
    torch.load(os.path.join(MSVD_model_weights_path,f'model_epoch_{epoch}.pth')))
model.eval()
pred = []
gt = []
pred_words = []
gt_words = []
with torch.no_grad():
    for src, tgt in load_test_src_tgt(MSVD_feature_path, backbone, batch_size):
        src = src.to(device, non_blocking=True)  # 将数据移到设备上

        decode_result, _ = beam_search(model=model, src=src,
                                        max_len=20, pad=PAD_ID, bos=BOS_ID, eos=EOS_ID,
                                        beam_size=5, device=device)
        final_sentences_ids = [h[0] for h in decode_result]  # 返回一个batch中每个样本的最佳结果

        with h5py.File(os.path.join(captions_h5_path,'MSVD.h5'), 'r') as f:  # 在迭代时才打开文件
            captions = f['captions']['test']
            
            batch_tgt_captions = [] # 获取每个批量视频的captios列表
            for video_tuple in tgt: # 先获取元组，因为每个batch的tgt被封装成一个元组然后存在一个list中
                for video_id in video_tuple: # 然后遍历元组中的元组，获取每个video_id
                    captions_list = []
                    single_vide_caption = captions[video_id] # 获取该video_id对应的captions列表
                    for caption in single_vide_caption: # 对一个视频的每条caption进行处理
                        captions_list.append(caption.tolist())
                    batch_tgt_captions.append(captions_list) # 把每个video_id的对应captions列表装入批量列表中

        batch_outputs = final_sentences_ids
        pred += batch_outputs
        gt += batch_tgt_captions

    for i in range(len(pred)):
        interval_tgt_words = []
        for j in range(len(gt[i])):
            tgt_words = tokenizer.decode(gt[i][j],skip_special_tokens=True)
            assert len(gt[i][j]) != 0,'len_gt error'
            # tgt_words = ' '.join(tgt_words)
            interval_tgt_words.append(tgt_words)
        out_words = tokenizer.decode(pred[i],skip_special_tokens=True)
        # out_words = ' '.join(out_words)
        
        pred_words.append(out_words)
        gt_words.append(interval_tgt_words)
        # print(gt_words)


average_rate = calculate_average_inclusion_rate(pred_words, gt_words)
print(f"Average Inclusion Rate: {average_rate}")
