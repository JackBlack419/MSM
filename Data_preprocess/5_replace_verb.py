import string
import os
import re
import sys
import torch
import h5py
import json
from nltk.tokenize import word_tokenize
from nltk import pos_tag
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

# 定义一个函数来识别并替换动词为 [MASK]
def replace_verbs(text):
    # 分词
    tokens = word_tokenize(text)
    # 词性标注
    #Tagged: [('The', 'DT'), ('cat', 'NN'), ('is', 'VBZ'), ('sitting', 'VBG'), ('on', 'IN'), ('the', 'DT'), ('mat', 'NN'), ('and', 'CC'), ('watching', 'VBG'), ('the', 'DT'), ('birds', 'NNS'), ('fly', 'NN'), ('.', '.')]
    tagged = pos_tag(tokens)

    # 替换动词为 [MASK], 在bert的词表中的id是103
    # 常见的动词标签包括: 'VB' (动词原形), 'VBD' (动词过去式), 'VBG' (动词现在分词/动名词), 'VBN' (动词过去分词), 'VBP' (动词非第三人称单数现在时), 'VBZ' (动词第三人称单数现在时)
    replaced_sentence = ' '.join(['[MASK]' if tag.startswith('VB') else token for token, tag in tagged])
    
    return replaced_sentence

# 加载bert的分词器
tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_path,max_len=20,truncation=True,return_tensors='pt')

MSRVTT_caption_json_list = ['MSRVTT_train_captions.json','MSRVTT_valid_captions.json']
MSVD_caption_json_list = ['MSVD_train_captions.json','MSVD_valid_captions.json']

# 预处理caption并保存到HDF5
def process_and_save_captions(json_list, dataset_name):
    hdf5_path = os.path.join(opt.captions_path,f'{dataset_name}.h5')
    # 打开之前用来存储数据的组
    with h5py.File(hdf5_path, 'a') as h5_file:
        captions_grp = h5_file.create_group('verb_q')  
        for json_filename in json_list:
            with open(os.path.join(captions_json_path, json_filename), 'r', encoding='utf-8') as json_file:
                captions_dict = json.load(json_file)['captions']
                
            for video_id, captions in captions_dict.items():
                # 去除captions中的标点符号
                captions = [replace_punctuation_with_space(caption) for caption in captions]
                # 替换captions中的动词为<MASK>
                captions = [replace_verbs(caption) for caption in captions]
                # caption词编码
                captions_encodeing = []
                for caption in captions:
                    # print(caption) # a car [MASK] [MASK]
                    caption = tokenizer.encode(caption,add_special_tokens=True,max_length=20,pad_to_max_length=True,return_tensors='pt',return_attention_mask=False)
                    # print(caption)
                    # input("Press Enter to continue...")
            #         tensor([[ 101, 1037, 2482,  103,  103,  102,    0,    0,    0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0,    0]])
                    captions_encodeing.extend(caption)
                # 为每个视频ID创建一个数据集来存储caption
                captions_grp.create_dataset(video_id, data=torch.stack(captions_encodeing))


# 处理并保存MSRVTT和MSVD的caption
process_and_save_captions(MSRVTT_caption_json_list, 'MSRVTT')
process_and_save_captions(MSVD_caption_json_list, 'MSVD')