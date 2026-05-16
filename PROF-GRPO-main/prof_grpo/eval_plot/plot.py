import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

def simplify_model_name(model_name):
    """
    将长模型名称简化为更易读的标签，同时保留关键信息。
    例如: 'Qwen2.5-Math-1.5B-grpo-numina_math-prm4-n8-gen-posgapmean-clip0.2'
    简化为: '1.5B-prm-posgapmean-clip0.2'
    """
    # 提取模型大小 (e.g., 1.5B, 7B)
    size_match = re.search(r'(\d+\.?\d*[Bb])', model_name)
    size = size_match.group(1) if size_match else 'UnknownSize'

    # 提取主要方法
    method = "UnknownMethod"
    if 'prm4-n8-gen-posgapmean' in model_name:
        method = 'prm-posgapmean'
    elif 'prm4-unc0.-n8-posvalmean' in model_name:
        method = 'prm-posvalmean'
    elif 'grpo-numina_math-n4' in model_name:
        method = 'grpo-n4'
    
    # 组合成新的简化名称
    return f"{size}-{method}"

def plot_filtered_model_curves(file_path):
    """
    读取CSV文件，筛选出模型名精确包含'cliph0.2'或'clip0.2'且temperature为0的模型，
    为每个符合条件的模型计算col1-col3的平均值，并在同一张图上绘制各自的性能曲线。

    参数:
    file_path (str): CSV文件的路径。
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 定义正则表达式以精确匹配 'clip0.2' 或 'cliph0.2'
        regex_pattern = r'clip(h?)0\.2([^0-9]|$)'
        cliph=0.2
        
        # 定义筛选的温度
        target_temperature = 1

        # 应用两个筛选条件
        df_filtered = df[
            (df['model'].str.contains(regex_pattern, na=False, regex=True)) &
            (df['temperature'] == target_temperature)
        ].copy()

        if df_filtered.empty:
            print("错误：在文件中没有找到同时满足筛选条件的模型。")
            return
            
        # 应用名称简化函数创建新的标签列
        df_filtered['short_name'] = df_filtered['model'].apply(simplify_model_name)

        # 计算col1, col2, 和 col3 的平均值
        df_filtered['average'] = df_filtered[['col1', 'col2', 'col3']].mean(axis=1)

        # --- 绘图 ---
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(14, 8))

        sns.lineplot(
            data=df_filtered,
            x='step',
            y='average',
            hue='short_name',  # 使用简化后的名称作为图例
            marker='o',
            linewidth=2.5,
            markersize=8
        )

        # 添加图表标题和坐标轴标签 (英文)
        title_text = f'Model Performance Comparison (clip{cliph}, temp={target_temperature})'
        plt.title(title_text, fontsize=18, weight='bold')
        plt.xlabel('Steps', fontsize=14)
        plt.ylabel('Average Accuracy', fontsize=14)
        
        # 优化图例
        plt.legend(title='Model', fontsize=11)
        
        # 优化网格和布局
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()


        # --- 动态生成文件名 ---
        base_name = os.path.splitext(file_path)[0]
        clip_str = f"clip{cliph}"
        temp_str = f"temp{target_temperature}"
        plot_filename = f'plots/{base_name}_{clip_str}_{temp_str}.png'

        # 保存图表
        plt.savefig(plot_filename, bbox_inches='tight')
        
        # 显示图表 (在本地运行时取消注释下面这行)
        # plt.show()

        print(f"图表已成功保存为 '{plot_filename}'")
        
        # 打印出每个模型的平均性能，以便快速预览
        print("\n符合条件的模型数据预览:")
        for model_name, group in df_filtered.groupby('short_name'):
            print(f"\n--- 模型: {model_name} ---")
            print(group[['step', 'average']].head())


    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。请确保文件路径正确。")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")

# --- 主程序入口 ---
if __name__ == '__main__':
    # 您上传的CSV文件的路径
    csv_file = 'Qwen2.5-Math-1.5B_data.csv'
    plot_filtered_model_curves(csv_file)
