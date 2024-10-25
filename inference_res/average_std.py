import os
import json
import statistics

def calculate_first_column_stats(matrix, idx):
    """
    计算一个 5 行 4 列的列表中每一行第一列的平均值和标准差

    :param matrix: 5 行 4 列的二维列表
    :return: (平均值, 标准差)
    """
    # 提取第idx列
    column = [row[idx]*100 for row in matrix]

    # 计算平均值和标准差
    mean_column = statistics.mean(column)
    std_dev_column = statistics.stdev(column)

    return mean_column, std_dev_column

# 设置你的文件夹路径
folder_path = '/media/disk6t/wlp/PycharmProjects/ECMLA/inference_res/verb/VIT/MSVD'
# 创建一个 5 行 4 列的列表，所有元素初始化为 0
rows, cols = 5, 4
matrix = [[0 for _ in range(cols)] for _ in range(rows)]

# 初始化存储每个分数的最后一个分值的列表
bleu_last_values = []
cider_last_values = []
meteor_last_values = []
rouge_last_values = []

# 遍历文件夹中的所有 JSON 文件
for idx in range(0,5):
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            # with open(os.path.join(folder_path, filename), 'r') as file:
            #     data = json.load(file)
            #     print(data)
            with open(os.path.join(folder_path, filename), 'r') as file:
                # for line in file:
                line = file.readlines()[idx]
                data = json.loads(line.strip())

                # 提取 BLEU 的最后一个分值
                bleu_last_values.append(data['BLEU'][-1])
                # 提取 CIDEr 的分值
                cider_last_values.append(data['CIDEr'])
                # 提取 METEOR 的分值
                meteor_last_values.append(data['METEOR'])
                # 提取 ROUGE 的分值
                rouge_last_values.append(data['ROUGE'])

    # # 找到每个分数类型的最大值
    # max_bleu_last_value = max(bleu_last_values)
    # max_cider_last_value = max(cider_last_values)
    # max_meteor_last_value = max(meteor_last_values)
    # max_rouge_last_value = max(rouge_last_values)
    # 找到每个指标的最大值
    max_values = [
        max(bleu_last_values),
        max(cider_last_values),
        max(meteor_last_values),
        max(rouge_last_values)
    ]
    bleu_last_values = []
    cider_last_values = []
    meteor_last_values = []
    rouge_last_values = []

    matrix[idx] = max_values

print(matrix)
blue4_mean, bleu4_std_dev = calculate_first_column_stats(matrix,idx=0)
CIDEr_mean, CIDEr_std_dev = calculate_first_column_stats(matrix,idx=1)
METEOR_mean, METEOR_std_dev = calculate_first_column_stats(matrix,idx=2)
ROUGE_mean, ROUGE_std_dev = calculate_first_column_stats(matrix,idx=3)

# 输出结果
print(f'BLEU4 最大值的平均值: {blue4_mean} 最大值的标准差{bleu4_std_dev}')
print(f'CIDEr 最大值的平均值: {CIDEr_mean} 最大值的标准差{CIDEr_std_dev}')
print(f'METEOR 最大值的平均值: {METEOR_mean} 最大值的标准差{METEOR_std_dev}')
print(f'ROUGE 最大值的平均值: {ROUGE_mean} 最大值的标准差{ROUGE_std_dev}')