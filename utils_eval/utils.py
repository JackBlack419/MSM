from torch.utils.data import IterableDataset, DataLoader
import torch
import torch.nn as nn
import h5py
from collections import OrderedDict, defaultdict
from utils_eval.coco_caption.pycocoevalcap.bleu.bleu import Bleu
from utils_eval.coco_caption.pycocoevalcap.cider.cider import Cider
from utils_eval.coco_caption.pycocoevalcap.rouge.rouge import Rouge
from utils_eval.coco_caption.pycocoevalcap.meteor.meteor import Meteor
import random


# 使用迭代器来做dataset
class DynamicDataset(IterableDataset):
    def __init__(self, fea_h5_path, data_h5_path, backbone, pattern):
        self.fea_h5_path = fea_h5_path # 只存储路径，不立即打开文件
        self.data_h5_path = data_h5_path  # 只存储路径，不立即打开文件
        self.backbone = backbone
        self.pattern = pattern
    def __iter__(self):
        with h5py.File(self.fea_h5_path,'r') as fea_f:
            with h5py.File(self.data_h5_path, 'r') as f:  # 在迭代时才打开文件
                captions = f['captions'][self.pattern]
                video_faetures = fea_f[self.backbone][self.pattern]
                used_caption = {i: set() for i in captions}  # 记录每一个视频的哪些caption已经使用过了

                # 遍历组中的所有数据集
                video_list = list(video_faetures.keys())  # 获取所有视频的名称列表
                while video_list:
                    for video_id in video_list:
                        un_used_caption = [i for i in range(len(captions[video_id])) if i not in used_caption[video_id]]
                        if un_used_caption:
                            video_feature = video_faetures[video_id][:]
                            caption_index = random.choice(un_used_caption)
                            caption = captions[video_id][caption_index]
                            used_caption[video_id].add(caption_index)
                            yield video_feature, caption
                            
                        else:  # 如果当前video的所有captions都已经使用，则不再遍历这个video
                            video_list.remove(video_id)

# with_verb模块的dataset
class Verb_DynamicDataset(IterableDataset):
    def __init__(self, fea_h5_path, data_h5_path, backbone, pattern):
        self.fea_h5_path = fea_h5_path # 只存储路径，不立即打开文件
        self.data_h5_path = data_h5_path
        self.backbone = backbone
        self.pattern = pattern

    def __iter__(self):
        with h5py.File(self.fea_h5_path,'r') as fea_f:
            with h5py.File(self.data_h5_path, 'r') as f:  # 在迭代时才打开文件
                video_faetures = fea_f[self.backbone][self.pattern]
                verb_tgt = f['verb_tgt']
                verb_q = f['verb_q']
                captions = f['captions'][self.pattern]
                used_caption = {i: set() for i in captions}  # 记录每一个视频的哪些caption已经使用过了

                # 遍历组中的所有数据集
                video_list = list(video_faetures.keys())  # 获取所有视频的名称列表
                while video_list:
                    for video_id in video_list:
                        un_used_caption = [i for i in range(len(captions[video_id])) if i not in used_caption[video_id]]
                        if un_used_caption:
                            video_feature = video_faetures[video_id][:]
                            caption_index = random.choice(un_used_caption)
                            caption = captions[video_id][caption_index]
                            v_q = verb_q[video_id][caption_index]
                            v_tgt = verb_tgt[video_id][caption_index]
                            used_caption[video_id].add(caption_index)
                            yield video_feature, caption, v_q, v_tgt
                        
                        else:  # 如果当前video的所有captions都已经使用，则不再遍历这个video
                            video_list.remove(video_id)

# with_noun模块的dataset
class Noun_DynamicDataset(IterableDataset):
    def __init__(self, fea_h5_path, data_h5_path, backbone, pattern):
        self.fea_h5_path = fea_h5_path # 只存储路径，不立即打开文件
        self.data_h5_path = data_h5_path
        self.backbone = backbone
        self.pattern = pattern

    def __iter__(self):
        with h5py.File(self.fea_h5_path,'r') as fea_f:
            with h5py.File(self.data_h5_path, 'r') as f:  # 在迭代时才打开文件
                video_faetures = fea_f[self.backbone][self.pattern]
                noun_tgt = f['noun_tgt']
                noun_q = f['noun_q']
                captions = f['captions'][self.pattern]
                used_caption = {i: set() for i in captions}  # 记录每一个视频的哪些caption已经使用过了

                # 遍历组中的所有数据集
                video_list = list(video_faetures.keys())  # 获取所有视频的名称列表
                while video_list:
                    for video_id in video_list:
                        un_used_caption = [i for i in range(len(captions[video_id])) if i not in used_caption[video_id]]
                        if un_used_caption:
                            video_feature = video_faetures[video_id][:]
                            caption_index = random.choice(un_used_caption)
                            caption = captions[video_id][caption_index]
                            n_q = noun_q[video_id][caption_index]
                            n_tgt = noun_tgt[video_id][caption_index]
                            used_caption[video_id].add(caption_index)
                            yield video_feature, caption, n_q, n_tgt
                        
                        else:  # 如果当前video的所有captions都已经使用，则不再遍历这个video
                            video_list.remove(video_id)

class integrated_module_dataset(IterableDataset):
    def __init__(self, fea_h5_path, data_h5_path, backbone, pattern):
        self.fea_h5_path = fea_h5_path # 只存储路径，不立即打开文件
        self.data_h5_path = data_h5_path  # 只存储路径，不立即打开文件
        self.backbone = backbone
        self.pattern = pattern

    def __iter__(self):
        with h5py.File(self.fea_h5_path,'r') as fea_f:
            with h5py.File(self.data_h5_path, 'r') as f:  # 在迭代时才打开文件
                video_faetures = fea_f[self.backbone][self.pattern]
                verb_tgt = f['verb_tgt']
                verb_q = f['verb_q']
                noun_tgt = f['noun_tgt']
                noun_q = f['noun_q']
                captions = f['captions'][self.pattern]
                used_caption = {i: set() for i in captions}  # 记录每一个视频的哪些caption已经使用过了

                # 遍历组中的所有数据集
                video_list = list(video_faetures.keys())  # 获取所有视频的名称列表
                while video_list:
                    for video_id in video_list:
                        un_used_caption = [i for i in range(len(captions[video_id])) if i not in used_caption[video_id]]
                        if un_used_caption:
                            video_feature = video_faetures[video_id][:]
                            caption_index = random.choice(un_used_caption)
                            caption = captions[video_id][caption_index]
                            v_q = verb_q[video_id][caption_index]
                            v_tgt = verb_tgt[video_id][caption_index]
                            n_q = noun_q[video_id][caption_index]
                            n_tgt = noun_tgt[video_id][caption_index]
                            used_caption[video_id].add(caption_index)
                            yield video_feature, caption, v_q, v_tgt, n_q, n_tgt
                            
                        else:  # 如果当前video的所有captions都已经使用，则不再遍历这个video
                            video_list.remove(video_id)

# 使用迭代器来做dataset
class Test_DynamicDataset(IterableDataset):
    def __init__(self, feature_h5_path, backbone):
        self.feature_h5_path = feature_h5_path
        self.backbone = backbone
    def __iter__(self):
        with h5py.File(self.feature_h5_path,'r') as fea_f:
            video_faetures = fea_f[self.backbone]['test']
            video_list = list(video_faetures.keys())  # 获取所有视频的名称列表
            for video_id in video_list:
                video_feature = video_faetures[video_id][:]
                yield video_feature, [video_id] # 返回视频特征和视频名   

def load_src_tgt(fea_h5_path, data_h5_path, backbone, pattern, batch_size):
    dataset = DynamicDataset(fea_h5_path, data_h5_path, backbone, pattern)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,num_workers=0)

    return dataloader

# with_verb模块的dataset
def verb_dataloader(fea_h5_path, data_h5_path, backbone, pattern, batch_size):
    dataset = Verb_DynamicDataset(fea_h5_path, data_h5_path, backbone, pattern)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,num_workers=0)

    return dataloader   

# With_noun模块的dataloader
def noun_dataloader(fea_h5_path, data_h5_path, backbone, pattern, batch_size):
    dataset = Noun_DynamicDataset(fea_h5_path, data_h5_path, backbone, pattern)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,num_workers=0)

    return dataloader 

def integrated_module_dataloader(fea_h5_path, data_h5_path, backbone, pattern, batch_size):
    dataset = integrated_module_dataset(fea_h5_path, data_h5_path, backbone, pattern)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,num_workers=0)

    return dataloader

def load_test_src_tgt(feature_h5_path, backbone, batch_size):
    dataset = Test_DynamicDataset(feature_h5_path, backbone)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return dataloader
    

# 设置损失函数
class LabelSmoothing(nn.Module):
    """标签平滑处理"""

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        loss = self.criterion(x, true_dist)

        return loss

def contrastive_similarity_loss(similarity_matrix, temperature=0.07):
    """
    对比相似度损失函数，鼓励同一位置的特征向量有更高的相似度。

    参数:
    - similarity_matrix (Tensor): 形状为 (bs, seq_len, seq_len) 的相似度矩阵
    - temperature (float): 温度系数，用于调整损失函数的平滑程度

    返回:
    - Tensor: 损失值
    """
    batch_size, seq_len, _ = similarity_matrix.size()

    # 将相似度矩阵按行转换为概率分布
    exp_sim_matrix = torch.exp(similarity_matrix / temperature)
    # 计算每行的概率分布之和
    exp_sim_sum = torch.sum(exp_sim_matrix, dim=-1, keepdim=True)
    # 归一化得到概率分布
    prob_matrix = exp_sim_matrix / exp_sim_sum

    # 计算对角线上每个元素的负对数似然
    diagonal_elements = torch.diag_embed(torch.diagonal(prob_matrix, dim1=-2, dim2=-1))
    log_prob_diagonal = -torch.log(diagonal_elements + 1e-8)

    # 计算平均损失
    loss = torch.mean(log_prob_diagonal)

    return loss

#测评函数
def language_eval(sample_seqs, groundtruth_seqs):
    assert len(sample_seqs) == len(groundtruth_seqs), 'length of sampled seqs is different from that of groundtruth seqs!'

    references, predictions = OrderedDict(), OrderedDict()
    for i in range(len(groundtruth_seqs)):
        references[i] = [groundtruth_seqs[i][j] for j in range(len(groundtruth_seqs[i]))]
    for i in range(len(sample_seqs)):
        predictions[i] = [sample_seqs[i]]

    predictions = {i: predictions[i] for i in range(len(sample_seqs))}
    references = {i: references[i] for i in range(len(groundtruth_seqs))}

    avg_bleu_score, bleu_score = Bleu(4).compute_score(references, predictions)
    print('avg_bleu_score == ', avg_bleu_score)
    avg_cider_score, cider_score = Cider().compute_score(references, predictions)
    print('avg_cider_score == ', avg_cider_score)
    m = Meteor()
    avg_meteor_score, meteor_score = m.compute_score(references, predictions)
    m.__exit__()
    print('avg_meteor_score == ', avg_meteor_score)
    avg_rouge_score, rouge_score = Rouge().compute_score(references, predictions)
    print('avg_rouge_score == ', avg_rouge_score)

    return {'BLEU': avg_bleu_score, 'CIDEr': avg_cider_score, 'METEOR': avg_meteor_score, 'ROUGE': avg_rouge_score}