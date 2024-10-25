import os
import re
import sys
import json
import string
import statistics
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# 获取当前文件的绝对路径
current_path = os.path.abspath(__file__)

# 获取当前文件所在目录的父目录的绝对路径
parent_dir = os.path.dirname(os.path.dirname(current_path))

# 将父目录添加到sys.path中，以便能够导入父目录中的config.py
sys.path.append(parent_dir)

# 从config.py文件中导入Config类
from config import Config
opt = Config()

# 定义存储标注文件路径的变量
captions_json_path = opt.captions_path

# 定义包含MSRVTT和MSVD数据集标注文件名的列表
MSRVTT_caption_json_list = ['MSRVTT_test_captions.json', 'MSRVTT_train_captions.json', 'MSRVTT_valid_captions.json']
MSVD_caption_json_list = ['MSVD_test_captions.json', 'MSVD_train_captions.json', 'MSVD_valid_captions.json']

# 定义一个函数来替换标点符号为空格
def replace_punctuation_with_space(text):
    # 创建翻译表，将标点符号映射为空格
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    # 使用 translate 方法替换标点符号为空格
    text_with_spaces = text.translate(translator)
    
    # 使用正则表达式将两个及以上的连续空格替换成一个空格
    cleaned_text = re.sub(r'\s{2,}', ' ', text_with_spaces)
    
    return cleaned_text

# 定义函数用于统计标注长度的分布
def count_captions_distribution(json_list):
    total_len_list = []
    for json_filename in json_list:
        with open(os.path.join(captions_json_path, json_filename), 'r', encoding='utf-8') as json_file:
            captions_dict = json.load(json_file)['captions']

        for video_id, captions in captions_dict.items():
            # 去除captions中的标点符号
            captions = [replace_punctuation_with_space(caption) for caption in captions]
            # 获取清洗后的标注长度列表
            length_list = [len(caption.split()) for caption in captions]
            # 将长度列表添加到总列表中
            total_len_list.append(length_list)

    return total_len_list[0],total_len_list[1],total_len_list[2]

# 调用函数分别获取MSRVTT和MSVD数据集的标注长度分布
MSRVTT_test_len_count,MSRVTT_train_len_count,MSRVTT_valid_len_count = count_captions_distribution(MSRVTT_caption_json_list)
MSVD_test_len_count,MSVD_train_len_count,MSVD_valid_len_count = count_captions_distribution(MSVD_caption_json_list)

# 寻找众数
def count_mode(len_list):
    freqs = Counter(len_list)  # 使用Counter统计每个元素的频率
    max_freq = max(freqs.values())  # 找到最高的频率
    return [num for num, freq in freqs.items() if freq == max_freq]  # 返回所有频率等于最高频率的元素,即众数，因为可能会有多众数

# 创建画布，设置子图布局
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))

# 在第一个子图上,使用seaborn绘制MSRVTT数据集的length统计箱线图
MSRVTT_len_data = MSRVTT_train_len_count + MSRVTT_test_len_count + MSRVTT_valid_len_count
sns.boxplot(y=MSRVTT_len_data, width=0.4, ax=axes[0, 0])

# 在第二个子图上,使用seaborn绘制MSVD数据集的length统计箱线图
MSVD_len_data = MSVD_train_len_count + MSVD_test_len_count + MSVD_valid_len_count
sns.boxplot(y=MSVD_len_data, width=0.4, ax=axes[0, 1])

# 在第三个子图上我们绘制MSRVTT、MSVD的train、test、valid的length的中位数
len_list = [MSRVTT_train_len_count,MSRVTT_test_len_count,MSRVTT_valid_len_count,MSVD_train_len_count,MSVD_test_len_count,MSVD_valid_len_count]
median_len_list = [np.median(data) for data in len_list]
# print("长度分布中位数：",median_len_list) # [6.5, 8.0, 5.5, 8.0, 8.0, 6.0]
xticks_median = ['VTT_train','VTT_test','VTT_valid','MSVD_train','MSVD_test','MSVD_valid'] # 设置X轴上的刻度标签
sns.barplot(x=xticks_median, y=median_len_list, ax=axes[0, 2])
axes[0,2].set_ylabel('len_median')


# 在第四个子图上我们绘制MSRVTT、MSVD的train、test、valid的length的平均数
len_list = [MSRVTT_train_len_count,MSRVTT_test_len_count,MSRVTT_valid_len_count,MSVD_train_len_count,MSVD_test_len_count,MSVD_valid_len_count]
mean_len_list = [np.mean(data) for data in len_list]
# print("长度分布平均数：",mean_len_list) # [7.5, 8.7, 7.1, 8.289473684210526, 8.864864864864865, 7.051724137931035]
xticks_mean = ['VTT_train','VTT_test','VTT_valid','MSVD_train','MSVD_test','MSVD_valid'] # 设置X轴上的刻度标签
sns.barplot(x=xticks_mean, y=mean_len_list, ax=axes[1, 0])
axes[1, 0].set_ylabel('len_mean')

# 在第五个子图上我们绘制MSRVTT、MSVD的train、test、valid的length的众数
len_list = [MSRVTT_train_len_count,MSRVTT_test_len_count,MSRVTT_valid_len_count,MSVD_train_len_count,MSVD_test_len_count,MSVD_valid_len_count]
xticks = ['VTT_train','VTT_test','VTT_valid','MSVD_train','MSVD_test','MSVD_valid'] # 设置X轴上的刻度标签
mode_len_list = []
xticks_mode = []
for item,name in zip(len_list,xticks):
    mode_list = count_mode(item)
    mode_len_list = mode_len_list + mode_list
    for i in range(len(mode_list)): # 每一个集合有多少众数，就添加多少次对于的xtick
        xticks_mode = xticks_mode + [name]
# print(f'长度分布众数: {mode_len_list}') # [6, 7, 8, 4, 9, 8, 8, 5]
sns.barplot(x=xticks_mode, y=mode_len_list, ax=axes[1, 1])
axes[1, 1].set_ylabel('len_mode')

# 在第六个子图上我们绘制MSRVTT、MSVD的train、test、valid的length的方差
len_list = [MSRVTT_train_len_count,MSRVTT_test_len_count,MSRVTT_valid_len_count,MSVD_train_len_count,MSVD_test_len_count,MSVD_valid_len_count]
var_len_list = [statistics.variance(data) for data in len_list]
# print("长度分布方差：",var_len_list)
xticks_var = ['VTT_train','VTT_test','VTT_valid','MSVD_train','MSVD_test','MSVD_valid'] # 设置X轴上的刻度标签
# print(f'长度分布方差: {var_len_list}') # [4.473684210526316, 10.43157894736842, 14.726315789473684, 5.292318634423897, 9.953453453453454, 11.69903206291591]
sns.barplot(x=xticks_var, y=var_len_list, ax=axes[1, 2])
axes[1, 2].set_ylabel('len_var')

# 设置每一个图像的标题
titles = ['MSRVTT_len_Boxplot','MSVD_len_Boxplot','len_median','len_mean','len_mode','len_var']
for ax, title in zip(axes.flat, titles):
    ax.set_title(title)

# 调整子图间距
plt.tight_layout()

# 显示和保存图像
plt.savefig('multiplot_figure.png', dpi=300)
plt.show()