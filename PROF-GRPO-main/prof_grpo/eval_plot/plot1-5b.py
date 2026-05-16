import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_specific_model_curves(file_path, model_config, temp_to_plot):
    """
    读取CSV文件，为指定的模型和温度绘制性能曲线，
    并使用用户自定义的简称和颜色。

    参数:
    file_path (str): CSV文件的路径。
    model_config (dict): 一个字典，键是模型的完整名称，
                         值是一个包含(简称, 颜色)的元组。
    temp_to_plot (float): 要筛选的温度值。
    """
    try:
        # --- 1. 读取和筛选数据 ---
        df = pd.read_csv(file_path)

        # 从配置字典中提取出模型全名、简称映射和颜色映射
        model_full_names = model_config.keys()
        short_name_map = {k: v[0] for k, v in model_config.items()}
        color_palette = {v[0]: v[1] for k, v in model_config.items()}

        # 筛选出在字典的键中指定的模型，并且温度匹配
        df_filtered = df[
            (df['model'].isin(model_full_names)) &
            (df['temperature'] == temp_to_plot)
        ].copy()

        if df_filtered.empty:
            print(f"错误：在文件 '{file_path}' 中没有找到任何指定的模型（在温度={temp_to_plot}下）。")
            print("请检查 'models_to_plot' 字典中的名称是否与CSV文件中的完全一致。")
            return
            
        # --- 2. 计算平均值并应用自定义简称 ---
        df_filtered['average'] = df_filtered[['col1', 'col2', 'col3']].mean(axis=1)
        # 使用 .map() 方法将字典中的自定义简称应用到新列
        df_filtered['short_name'] = df_filtered['model'].map(short_name_map)

        # --- 3. 绘图 ---
        sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#f7f7f7", "grid.linestyle": "--"})
        plt.figure(figsize=(14, 8))

        legend_order = [v[0] for v in model_config.values()]

        # 使用 palette 参数来指定每条线的颜色
        sns.lineplot(
            data=df_filtered,
            x='step',
            y='average',
            hue='short_name',  # 使用包含自定义简称的列
            hue_order=legend_order, # 确保图例顺序与配置字典一致
            palette=color_palette, # 传入自定义的颜色字典
            marker='o',
            linewidth=2.5,
            markersize=8
        )

        # --- 4. 设置图表样式和标题 ---
        # title_text = f'1.5B Model Performance Comparison (temp={temp_to_plot})'
        # plt.title(title_text, fontsize=18, weight='bold')
        plt.axhline(y=0.441, color='#1f77b4', linestyle='--', linewidth=2, label='GRPO')

        plt.xlabel('Training Steps', fontsize=30)
        plt.ylabel('Average Accuracy', fontsize=30)
        
        # 设置x轴和y轴刻度字体大小
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        
        # 自定义图例
        plt.legend(title='Model', fontsize=25, loc='lower right')
        
        plt.grid(True, which='both', linestyle=':', linewidth=1)
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
    # models_to_plot = {# main models
    #     'Qwen2.5-Math-1.5B-grpo-numina_math-n4-cliph0.28': ('GRPO', '#1f77b4'),
    #     #'Qwen2.5-Math-1.5B-grpo-numina_math-prm4-unc0.-n8-posvalmean-clip0.28': ('PROF-GRPO', '#2ca02c'),
    #     'Qwen2.5-Math-1.5B-grpo-numina_math-n4-beta0.8-plus_Qwen-clip0.2': ('Blend-PRM-GRPO', "#d68a27"),
    #     'Qwen2.5-Math-1.5B-grpo-numina_math-prm4-n8-Qwen-sepval-clip0.28': ('PROF-GRPO', "#bf3415"),
    # }

    models_to_plot = { #2. + - prm filter
        'Qwen2.5-Math-1.5B-grpo-numina_math-prm4-n8-Qwen-sepval-clip0.28': ('PROF-GRPO', "#bf3415"),
        'Qwen2.5-Math-1.5B-grpo-numina_math-prm4-unc0.-n8-posvalmean-clip0.28': ('Filter-Correct', "#7E24D9"),
        'Qwen2.5-Math-1.5B-grpo-numina_math-prm4-n8-Qwen-negval-clip0.28': ('Filter-Incorrect', "#b9822f"),
        #'Qwen2.5-Math-1.5B-grpo-numina_math-prm0.5-unc0.-n8-ratio-posvalmean-clip0.2': ('Ratio-Correct', "#CDBC28"),
        #'Qwen2.5-Math-1.5B-grpo-numina_math-prm0.5-unc0.-n8-ratio-posvalmean': 'Qwen-PRM-ratio-pos-valmean-cliph0.28',
        #'Qwen2.5-Math-1.5B-grpo-numina_math-prm0.5-unc0.-n8-ratio-sepvalmean': 'Qwen-PRM-ratio-sep-valmean-cliph0.28',
        #'Qwen2.5-Math-1.5B-grpo-numina_math-prm0.5-ent_coef0.05-n8-sepvalmean': 'Qwen-PRM-0.05ent-sep-valmean-cliph0.2',
        'Qwen2.5-Math-1.5B-grpo-numina_math-prm4-n8-random-base-clip0.28': ('Filter-Random', "#CDBC28"),
        'Qwen2.5-math-1.5B-grpo-numina_math-prm4-n8-Qwen-nsep-mean-cliph0.28': ('w/o Separation', "#4DBEC4"),
    }

    # models_to_plot = { # n rollout
    #     'Qwen2.5-Math-1.5B-grpo-numina_math-prm4-unc0.-n8-posvalmean-clip0.28': ('PROF-GRPO-n8', '#2ca02c'),
    #     'Qwen2.5-Math-1.5B-grpo-numina_math-prm12-n16-Qwen-posval-clip0.28': ('PROF-GRPO-n16', "#24a6c7"),
    #     'Qwen2.5-Math-1.5B-grpo-numina_math-prm8-n12-Qwen-posval-clip0.28': ('PROF-GRPO-n16', "#3549AD"),
    # }

    # models_to_plot = { # variants
    #     'Qwen2.5-Math-1.5B-grpo-numina_math-prm4-unc0.-n8-posvalmean-clip0.28': 'Qwen-PRM-filter-pos-valmean-cliph0.28',
    #     'Qwen2.5-Math-1.5B-grpo-numina_math-prm0.5-unc0.-n8-ratio-posvalmean-clip0.2': 'Qwen-PRM-filter-ratio-pos-valmean-cliph0.2',
    #     #'Qwen2.5-Math-1.5B-grpo-numina_math-prm0.5-unc0.-n8-ratio-posvalmean': 'Qwen-PRM-ratio-pos-valmean-cliph0.28',
    #     #'Qwen2.5-Math-1.5B-grpo-numina_math-prm0.5-unc0.-n8-ratio-sepvalmean': 'Qwen-PRM-ratio-sep-valmean-cliph0.28',
    #     #'Qwen2.5-Math-1.5B-grpo-numina_math-prm0.5-ent_coef0.05-n8-sepvalmean': 'Qwen-PRM-0.05ent-sep-valmean-cliph0.2',
    #     #'Qwen2.5-Math-1.5B-grpo-numina_math-Qwen-sep-prm4-ent_coef0.05-n8-clip0.28': 'Qwen-PRM-0.05ent-sep-valmean-cliph0.28',
    #     #'Qwen2.5-Math-1.5B-grpo-numina_math-Qwen-pos-prm4-ent_coef0.05-n8-clip0.28': 'Qwen-PRM-0.05ent-pos-valmean-cliph0.28',
    #     'Qwen2.5-math-1.5B-grpo-numina_math-prm4-n8-Qwen-posmin-cliph0.28': 'Qwen-PRM-pos-min-n8-cliph0.28',
    #     'Qwen2.5-math-1.5B-grpo-numina_math-prm4-n8-Qwen-nsep-mean-cliph0.28': 'Qwen-PRM-nsep-mean-n8-cliph0.28',
    # }
    
    # --- 指定要绘制的温度 ---
    temperature_to_plot = 1.0

    # --- 指定CSV文件路径 ---
    csv_file = 'Qwen2.5-Math-1.5B_data.csv'
    
    # 调用绘图函数
    plot_specific_model_curves(csv_file, models_to_plot, temperature_to_plot)

