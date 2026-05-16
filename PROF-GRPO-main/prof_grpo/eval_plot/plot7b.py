import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_specific_model_curves(file_path, model_mapping, temp_to_plot):
    """
    读取CSV文件，为指定的模型和温度绘制性能曲线，
    并使用用户自定义的简称作为图例。

    参数:
    file_path (str): CSV文件的路径。
    model_mapping (dict): 一个字典，键是模型的完整名称，值是自定义的简称。
    temp_to_plot (float): 要筛选的温度值。
    """
    try:
        # --- 1. 读取和筛选数据 ---
        df = pd.read_csv(file_path)

        # 筛选出在字典的键中指定的模型，并且温度匹配
        df_filtered = df[
            (df['model'].isin(model_mapping.keys())) &
            (df['temperature'] == temp_to_plot)
        ].copy()

        if df_filtered.empty:
            print(f"错误：在文件 '{file_path}' 中没有找到任何指定的模型（在温度={temp_to_plot}下）。")
            print("请检查 'models_to_plot' 字典中的名称是否与CSV文件中的完全一致。")
            return
            
        # --- 2. 计算平均值并应用自定义简称 ---
        df_filtered['average'] = df_filtered[['col1', 'col2', 'col3']].mean(axis=1)
        # 使用 .map() 方法将字典中的自定义简称应用到新列
        df_filtered['short_name'] = df_filtered['model'].map(model_mapping)

        # --- 3. 绘图 ---
        sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#f7f7f7", "grid.linestyle": "--"})
        plt.figure(figsize=(14, 8))

        sns.lineplot(
            data=df_filtered,
            x='step',
            y='average',
            hue='short_name',  # 使用包含自定义简称的列
            marker='o',
            linewidth=2.5,
            markersize=8
        )

        # --- 4. 设置图表样式和标题 ---
        title_text = f'7B Model Performance Comparison (temp={temp_to_plot})'
        plt.title(title_text, fontsize=18, weight='bold')
        plt.xlabel('Steps', fontsize=14)
        plt.ylabel('Average Accuracy', fontsize=14)
        
        # 自定义图例
        plt.legend(title='Model', fontsize=12)
        
        plt.grid(True, which='both', linestyle=':', linewidth=0.8)
        plt.tight_layout()

        # --- 5. 保存图表 ---
        base_name = os.path.splitext(file_path)[0]
        plot_filename = f'plots/{base_name}_temp{temp_to_plot}_comparison.png'
        
        # 确保 'plots' 文件夹存在
        os.makedirs('plots', exist_ok=True)

        plt.savefig(plot_filename, bbox_inches='tight')
        
        print(f"图表已成功保存为 '{plot_filename}'")
        
        print("\n已绘制模型的预览数据:")
        for model_name, group in df_filtered.groupby('short_name'):
            print(f"\n--- 模型: {model_name} ---")
            print(group[['step', 'average']].head())

    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。请确保文件路径正确。")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")

# --- 主程序入口 ---
if __name__ == '__main__':
    
    # --- 在这里指定您想要绘制的模型及其自定义简称 ---
    # 格式: { '在CSV中的完整模型名称': '您想在图例中显示的简称' }
    models_to_plot = {
        'Qwen2.5-Math-7B-grpo-numina_math-n4': 'GRPO-cliph0.2',
        'Qwen2.5-Math-78-grpo-numina_math-prm4-n8-gen-posgapmean-clip0.2': 'Gen-PRM-filter-pos-mean-cliph0.2',
        'Qwen2.5-Math-7B-grpo-numina_math-prm4-unc0.-n8-posvalmean-clip0.2': 'Qwen-PRM-filter-pos-valmean-cliph0.2',
        'Qwen2.5-Math-7B-grpo-numina_math-prm4-n8-Qwen-posval-clip0.24': 'Qwen-PRM-filter-pos-valmean-cliph0.24',
    }
    
    # --- 指定要绘制的温度 ---
    temperature_to_plot = 1.0

    # --- 指定CSV文件路径 ---
    csv_file = 'Qwen2.5-Math-7B_data.csv'
    
    # 调用绘图函数
    plot_specific_model_curves(csv_file, models_to_plot, temperature_to_plot)

