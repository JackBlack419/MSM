import string
import os
import re
import sys
import torch
import h5py
import json
# import nltk
from transformers import BertTokenizer

# 获取当前文件所在的绝对路径
current_path = os.path.abspath(__file__)

# 获取当前文件所在目录的父目录的绝对路径
parent_dir = os.path.dirname(os.path.dirname(current_path))

# 将父目录添加到sys.path中
sys.path.append(parent_dir)

# 现在可以从A目录中导入config.py
from config import Config
opt = Config()

bert_tokenizer_path = opt.BertTokenizer_path # 存储了bert分词器配置文件的路劲
captions_json_path = opt.captions_path # 存储了captions（JSON格式）的路路径
hdf5_save_path = opt.captions_path # 将captions进行trunc_pad处理后使用hdf5文件存储的存储路径,实际上和JSON文件存储在同一文件夹下

# 定义一个函数来替换标点符号为空格
def replace_punctuation_with_space(text):
    # 创建翻译表，将标点符号映射为空格
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    # 使用 translate 方法替换标点符号为空格
    text_with_spaces = text.translate(translator)
    
    # 使用正则表达式将两个及以上的连续空格替换成一个空格
    cleaned_text = re.sub(r'\s{2,}', ' ', text_with_spaces)
    
    return cleaned_text

# 加载bert的分词器
tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_path,max_len=20,truncation=True,return_tensors='pt')

MSRVTT_caption_json_list = ['MSRVTT_test_captions.json','MSRVTT_train_captions.json','MSRVTT_valid_captions.json']
MSVD_caption_json_list = ['MSVD_test_captions.json','MSVD_train_captions.json','MSVD_valid_captions.json']
split_list = ['test','train','valid']

# 预处理caption并保存到HDF5
def process_and_save_captions(json_name, dataset_name, mode):
    # 创建一个hdf5文件保存train，valid, test对应的captions
    hdf5_path = os.path.join(opt.captions_path,f'{dataset_name}.h5')
    with h5py.File(hdf5_path, 'a') as h5_file:
        if 'captions' not in h5_file:
            captions_grp = h5_file.create_group('captions')  # 创建一个组来存放所有caption数据集
        else:
            captions_grp = h5_file['captions']  # 如果组已经存在，则获取该组
            
        with open(os.path.join(opt.captions_path,json_name), 'r', encoding='utf-8') as json_file:
            captions_dict = json.load(json_file)['captions']

        sub_grp = captions_grp.create_group(mode) # 按照train,valid,test来存储captions
        for video_id, captions in captions_dict.items():
            # 去除captions中的标点符号
            captions = [replace_punctuation_with_space(caption) for caption in captions]
            # caption词编码
            captions_encodeing = []
            for caption in captions:
                # print(caption) # a band performing in a small club
                caption = tokenizer.encode(caption,add_special_tokens=True,max_length=20,pad_to_max_length=True,return_tensors='pt',return_attention_mask=False)
                # print(caption)
                # input("Press Enter to continue...")
                # tensor([[ 101, 1037, 2316, 4488, 1999, 1037, 2235, 2252,  102,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0]])
                captions_encodeing.extend(caption)
            # 为每个视频ID创建一个数据集来存储caption
            sub_grp.create_dataset(video_id, data=torch.stack(captions_encodeing))

# 处理并保存MSRVTT和MSVD的caption
for json_name,split_name in zip(MSRVTT_caption_json_list,split_list):
    process_and_save_captions(json_name, 'MSRVTT', split_name)
for json_name,split_name in zip(MSVD_caption_json_list,split_list):
    process_and_save_captions(json_name, 'MSVD', split_name)