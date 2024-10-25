import matplotlib.pyplot as plt

# 第一组数据
labels = ['baseline', 'with_noun', 'with_verb', 'full']
noun_values_1 = [81.85, 83.09, 83.12, 84.76]
verb_values_1 = [82.42, 83.72, 83.44, 84.50]

# 第二组数据
noun_values_2 = [82.13, 84.84, 83.46, 85.12]
verb_values_2 = [79.59, 81.68, 82.56, 83.16]

# 创建折线图
plt.figure(figsize=(12, 8))

# 绘制第一组数据的折线图
plt.plot(labels, noun_values_1, marker='o', linestyle='-', color='blue', label='MSRVTT - noun')
plt.plot(labels, verb_values_1, marker='o', linestyle='-', color='orange', label='MSRVTT - verb')

# 绘制第二组数据的折线图
plt.plot(labels, noun_values_2, marker='s', linestyle='--', color='blue', label='MSVD - noun')
plt.plot(labels, verb_values_2, marker='s', linestyle='--', color='orange', label='MSVD - verb')

# 添加标题和标签
plt.title('Model Performance Comparison')
plt.xlabel('Model Variants')
plt.ylabel('Performance (%)')

# 添加图例
plt.legend()

# 显示数值标签
for i, v in enumerate(noun_values_1):
    plt.text(i, v + 0.2, str(v), ha='center', color='blue')
for i, v in enumerate(verb_values_1):
    plt.text(i, v + 0.2, str(v), ha='center', color='orange')

for i, v in enumerate(noun_values_2):
    plt.text(i, v - 0.5, str(v), ha='center', color='blue')
for i, v in enumerate(verb_values_2):
    plt.text(i, v - 0.5, str(v), ha='center', color='orange')

# 保存图表为图像文件
plt.savefig('model_performance_comparison.png')

# 显示图表
plt.show()