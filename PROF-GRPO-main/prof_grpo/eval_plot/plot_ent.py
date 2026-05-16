import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import re

# 这是一个包含您的数据的多行字符串。
# 在实际应用中，您可能会从文件中读取它。
file_content = """
GRPO
[0.36, 0.29, 0.27, 0.26, 0.25, 0.24, 0.24, 0.23, 0.25, 0.26, 0.26, 0.25,0.24, 0.26, 0.27, 0.27, 0.29, 0.27, 0.29, 0.30, 0.34,0.37,0.39,0.39,0.43,0.45,0.55]
PROF-GRPO
[0.35,0.3,0.26,0.24,0.22,0.21,0.21,0.2,0.2,0.2,0.19,0.18,0.18,0.18,0.18,0.18,0.18,0.18,0.18,0.18,0.19,0.19,0.2,0.2,0.2,0.21,0.22,0.22,0.22]
Balance-Correct
[0.35,0.31,0.27,0.24, 0.23,0.23,0.22,0.21,0.21, 0.2, 0.2, 0.19, 0.19, 0.19, 0.19, 0.18, 0.19, 0.18, 0.18, 0.18, 0.18, 0.17, 0.18,0.17,0.17, 0.18, 0.18, 0.17,0.17]
Balance-Random
[0.35,0.3,0.27,0.25,0.25,0.24,0.24,0.23,0.24,0.24,0.23,0.23,0.22,0.22,0.22,0.21,0.22,0.21,0.21,0.21,0.22,0.22,0.23,0.22,0.22,0.22,0.23,0.23,0.24]
PROF-GRPO w/o Separation
[0.35,0.31,0.26,0.25,0.24,0.23,0.23,0.22,0.23,0.22,0.22,0.22,0.22,0.22,0.23,0.22,0.23,0.23,0.23,0.24,0.24,0.26,0.28,0.26,0.26,0.28,0.29,0.32,0.37]
"""

# 解析数据
lines = file_content.strip().split('\n')
all_data_for_df = []
current_experiment_label = None

for line in lines:
    line = line.strip()
    if not line:
        continue
    
    # 检查行是数据（以'['开头）还是标签
    if line.startswith('[') and line.endswith(']'):
        if current_experiment_label is None:
            continue  # 避免没有标签的数据
        
        # 清理字符串并将其转换为浮点数列表
        values_str = line.strip('[]')
        try:
            values = [float(v) for v in re.split(r',\s*', values_str)]
        except ValueError:
            print(f"警告：跳过标签 {current_experiment_label} 的格式错误数据")
            continue

        # 生成相应的步数值
        steps = [1 + i * 10 for i in range(len(values))]
        
        # 将此实验的数据附加到我们的主列表中
        for step, loss in zip(steps, values):
            all_data_for_df.append({
                'step': step,
                'Entropy Loss': loss,
                'Experiment': current_experiment_label
            })
        current_experiment_label = None  # 重置标签
    else:
        current_experiment_label = line

# 创建一个pandas DataFrame
df = pd.DataFrame(all_data_for_df)

# 使用seaborn创建图表
plt.figure(figsize=(14, 8))
sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#f7f7f7", "grid.linestyle": "--"})
ax = sns.lineplot(data=df, x='step', y='Entropy Loss', hue='Experiment', marker='o')

# 优化图表的标题和标签
#plt.title('不同实验的熵损失与步数关系图', fontsize=16)
plt.xlabel('Training Steps', fontsize=30)
plt.ylabel('Entropy Loss', fontsize=30)

# 调整图例以防止其与图表重叠
plt.legend(title='Model', fontsize=25)
plt.tight_layout() 

# 将图表保存到文件
plt.savefig("entropy_loss_curves.png")