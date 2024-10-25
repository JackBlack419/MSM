import json
import os
import sys


# 获取当前文件所在的绝对路径
current_path = os.path.abspath(__file__)

# 获取当前文件所在目录的父目录的绝对路径
parent_dir = os.path.dirname(os.path.dirname(current_path))

# 将父目录添加到sys.path中
sys.path.append(parent_dir)

# 现在可以从A目录中导入config.py
from config import Config
opt = Config()

raw_MSRVTT_captions_path = os.path.join(opt.raw_captions_path,'MSRVTT')
raw_MSVD_captions_path = os.path.join(opt.raw_captions_path,'MSVD')


MSRVTT_train_path = os.path.join(opt.captions_path,'MSRVTT_train_captions.json')
MSRVTT_valid_path = os.path.join(opt.captions_path,'MSRVTT_valid_captions.json')
MSRVTT_test_path = os.path.join(opt.captions_path,'MSRVTT_test_captions.json')
MSVD_train_path = os.path.join(opt.captions_path,'MSVD_train_captions.json')
MSVD_valid_path = os.path.join(opt.captions_path,'MSVD_valid_captions.json')
MSVD_test_path = os.path.join(opt.captions_path,'MSVD_test_captions.json')

# 处理MSRVTT的train,valid的raw_captions
with open(os.path.join(raw_MSRVTT_captions_path, 'train_val_videodatainfo.json'), 'r') as f1:
    train_valid_dict = {}
    train_valid_video_id_list = []
    raw_data = json.load(f1)
    sentences = raw_data['sentences']
    
    for sentence in sentences:
        video_id = sentence['video_id']
        caption = sentence['caption']
        
        if video_id not in train_valid_video_id_list:
            train_valid_video_id_list.append(video_id)
            train_valid_dict[video_id] = [caption]
        else:
            train_valid_dict[video_id].append(caption)

    with open(MSRVTT_train_path,'w') as f2:
        train_dict = {}
        start_id = 0
        end_id = start_id + 6513
        for id in range(start_id,end_id):
            caption = train_valid_dict[f'video{id}']
            train_dict[f'video{id}'] = caption
        train_captions = {'captions':train_dict}
        json.dump(train_captions,f2)
    with open(MSRVTT_valid_path,'w') as f3:
        valid_dict = {}
        start_id = 6513
        end_id = start_id + 497
        for id in range(start_id, end_id):
            caption = train_valid_dict[f'video{id}']
            valid_dict[f'video{id}'] = caption
        valid_captions = {'captions': valid_dict}
        json.dump(valid_captions,f3)

# 处理MSRVTT的test的raw_captions
with open(os.path.join(raw_MSRVTT_captions_path,'test_videodatainfo.json'),'r') as f1:
    test_dict = {}
    test_video_id_list = []
    raw_data = json.load(f1)
    sentences = raw_data['sentences']
    for sentence in sentences:
        video_id = sentence['video_id']
        caption = sentence['caption']
        if sentence['video_id'] not in test_video_id_list:
            test_video_id_list.append(video_id)
            test_dict[video_id] = [caption]
        else:
            test_dict[video_id].append(caption)

    test_captions = {'captions':test_dict}
    with open(MSRVTT_test_path,'w') as f2:
        json.dump(test_captions,f2)

# 处理MSVD的raw_captions
with open(os.path.join(raw_MSVD_captions_path,'captions.txt'),'r') as f1:
    MSVD_dict = {}
    video_id_list = []
    raw_data = f1.readlines()
    for line in raw_data:
        video_id, caption = line.rstrip('\n').split(' ',1)
        if video_id not in video_id_list:
            video_id_list.append(video_id)
            MSVD_dict[video_id] = [caption]
        else:
            MSVD_dict[video_id].append(caption)
    # 1. 获取字典的键列表
    keys = list(MSVD_dict.keys())

    # 2.分割键
    keys_1 = keys[:1200]
    keys_2 = keys[1200:1300]
    keys_3 = keys[1300:]

    # 3. 生成新的字典
    train_dict = {key: MSVD_dict[key] for key in keys_1}
    valid_dict = {key: MSVD_dict[key] for key in keys_2}
    test_dict = {key: MSVD_dict[key] for key in keys_3}
        
    with open(MSVD_train_path,'w') as f1:
        train_captions = {'captions':train_dict}
        json.dump(train_captions,f1)
    with open(MSVD_valid_path,'w') as f2:
        valid_captions = {'captions':valid_dict}
        json.dump(valid_captions,f2)
    with open(MSVD_test_path,'w') as f3:
        test_captions = {'captions':test_dict}
        json.dump(test_captions,f3)