import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 将您图片中的数据整理成一个pandas DataFrame
# Seaborn的lineplot最适合使用“长格式”数据，
# 其中一列用于x轴(n)，一列用于y轴(value)，一列用于区分曲线(category)。
data = {
    'n': [4, 8, 12, 16,  # n的值 for 'Both'
          4, 8, 12, 16],     # n的值 for 'Correct' (图片中没有n=8)
    'value': [0.499, 0.516, 0.488, 0.459,     # 对应的 'Both' 的值
              0.499, 0.509, 0.518, 0.494],        # 对应的 'Correct' 的值
    'category': ['Both', 'Both', 'Both','Both',
                 'Correct', 'Correct', 'Correct', 'Correct']
}

df = pd.DataFrame(data)

# 打印DataFrame，确认数据结构
print("创建的DataFrame:")
print(df)
print("-" * 25)


# 2. 使用Seaborn进行绘图
plt.figure(figsize=(13, 8))
sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#f7f7f7", "grid.linestyle": "--"})

# 绘制线图
# - x='n': 指定n列作为x轴
# - y='value': 指定value列作为y轴
# - hue='category': 根据category列的内容来区分并绘制不同的曲线
# - marker='o': 在每个数据点上添加一个圆形标记
# - style='category': 为不同曲线使用不同的线条样式（可选，但能增强区分度）
line_plot = sns.lineplot(
    data=df,
    x='n',
    y='value',
    hue='category',
    style='category', # 使用不同线条样式
    marker='o',       # 添加数据点标记
    markersize=8,     # 标记大小
    linewidth=2.5     # 线条宽度
)

# 3. 自定义图表标题和轴标签
#plt.title('Performance Comparison: Both vs. Correct', fontsize=16)
plt.xlabel('Rollout Size (n)', fontsize=30)
plt.ylabel('Average Accuracy', fontsize=30)

# 设置x轴刻度，确保它们是整数
plt.xticks([4, 8, 12, 16], fontsize=25)
plt.yticks(fontsize=25)

# 优化图例
plt.legend(title='Category', fontsize=25)

# 显示图表
plt.show()
plt.savefig('./plots/nroll_plot.png')  # 保存图表为PNG文件