import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from pathlib import Path

# --- 1. 配置区 (您只需要修改这里) ---

# 指定存放所有模型评估结果的根目录
# 脚本会在此目录下查找以模型全名命名的子文件夹
EVAL_OUTPUT_DIR = str(Path.home() / "PRM_filter/eval/output")

# 定义要计算平均分的基准测试项目
# 脚本会计算这些项目分数的平均值
BENCHMARKS_TO_AVERAGE = [
    'math500', 'minerva_math', 'olympiadbench',
] #, 'aime24', 'amc23'

# 定义要绘制的模型及其在图例中显示的简称
# 格式: { '模型完整名称 (即文件夹名)': '图例中显示的简称' }
# MODELS_TO_PLOT = { # main
#     'Qwen2.5-Math-7B-grpo-numina_math-n4': ('GRPO', '#1f77b4'),
#     #'Qwen2.5-Math-7B-grpo-numina_math-prm4-n8-Qwen-posval-clip0.24': ('PROF-GRPO-0.24', "#11532f"),
#     #'Qwen2.5-Math-7B-grpo-numina_math-prm4-n8-Qwen-posval-clip0.2': ('PROF-GRPO', '#2ca02c'),
#     'Qwen2.5-Math-7B-grpo-numina_math-prm4-n8-Qwen-sepval-clip0.2': ('PROF-GRPO', "#bf3415"),
#     'Qwen2.5-Math-7B-grpo-numina_math-n4-beta0.8-plus_Qwen-clip0.2': ('Blend-PRM-GRPO', "#d68a27"),
# }

MODELS_TO_PLOT = {
    'Qwen2.5-Math-7B-grpo-numina_math-prm4-n8-Qwen-sepval-clip0.2': ('PROF-GRPO', "#bf3415"),
    'Qwen2.5-Math-7B-grpo-numina_math-prm4-n8-Qwen-posval-clip0.2': ('Filter-Correct', "#7E24D9"),
    'Qwen2.5-Math-7B-grpo-numina_math-prm4-n8-Qwen-negval-clip0.2': ('Filter-Incorrect', "#b9822f"),
    'Qwen2.5-Math-7B-grpo-numina_math-prm4-n8-random-base-clip0.24': ('Filter-Random', "#CDBC28"),
    #'Qwen2.5-Math-7B-grpo-numina_math-prm0.5-unc0.-n8-ratio-posvalmean-clip0.2': ('Ratio-Correct', "#CDBC28"),
    'Qwen2.5-Math-7B-grpo-numina_math-prm4-n8-Qwen-nsep-mean-clip0.24': ('w/o Separation', "#4DBEC4"),
}

# MODELS_TO_PLOT = {
#     'Qwen2.5-Math-7B-grpo-numina_math-prm4-n8-Qwen-sepval-clip0.2': ('$n=8$', "#bf3415"),
#     #'Qwen2.5-Math-7B-grpo-numina_math-prm4-n8-Qwen-posval-clip0.2': ('Balance-Correct', "#7E24D9"),
#     'Qwen2.5-Math-7B-grpo-numina_math-prm8-n12-Qwen-sepval-clip0.2': ('n=12', "#cc7427"),
#     'Qwen2.5-Math-7B-grpo-numina_math-prm8-n12-Qwen-posval-clip0.2': ('n=12-pos', "#36ba45"),
#     'Qwen2.5-Math-7B-grpo-numina_math-prm12-n16-Qwen-posval-clip0.24': ('n=16-pos', "#27a2c7"),
# }

# 指定要绘制的温度和结果文件名
# 脚本会在每个模型文件夹下查找 temp{TEMP_TO_PLOT}_K16.jsonl
TEMP_TO_PLOT = 1.0
EVAL_FILENAME = f"temp{TEMP_TO_PLOT}_K16.jsonl"

# 图表和输出文件设置
PLOT_TITLE = f'7B Model Performance Comparison (temp={TEMP_TO_PLOT})'
OUTPUT_DIR = 'plots' # 保存图表的文件夹
OUTPUT_FILENAME = Path(OUTPUT_DIR) / f'7B_model_comparison_temp{TEMP_TO_PLOT}.png'

# --- 脚本主程序 (无需修改以下内容) ---

def process_evaluation_files(base_dir, model_mapping, benchmarks):
    """
    遍历指定目录，读取并处理所有在 model_mapping 中定义的模型的 JSONL 数据。
    """
    all_data = []
    print("开始处理评估文件...")

    for model_full_name in model_mapping.keys():
        jsonl_path = Path(base_dir) / model_full_name / EVAL_FILENAME

        if not jsonl_path.exists():
            print(f"  - 警告: 未找到文件 -> {jsonl_path}")
            continue
        
        print(f"  - 正在读取: {model_full_name}")
        with open(jsonl_path, 'r', encoding='utf8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    
                    # 提取所有指定基准测试的分数
                    scores = [data[bench]['score'] for bench in benchmarks if bench in data]
                    
                    # 如果有分数，则计算平均值
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        all_data.append({
                            'model_full': model_full_name,
                            'step': data['step'],
                            'average_accuracy': avg_score
                        })
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"    - 解析行时出错: {line.strip()} | 错误: {e}")

    if not all_data:
        return None

    # 将收集到的数据转换为Pandas DataFrame
    df = pd.DataFrame(all_data)
    print("\n数据处理完成。")
    return df

def create_comparison_plot(df, model_config):
    """
    使用处理好的DataFrame生成并保存对比图。
    """
    if df is None or df.empty:
        print("错误：没有可供绘图的数据。请检查文件路径和配置。")
        return
        
    print("开始生成图表...")
    
    # 从配置中提取简称映射和颜色映射
    short_name_map = {k: v[0] for k, v in model_config.items()}
    color_palette = {v[0]: v[1] for k, v in model_config.items()}

    # 应用简称
    df['Model'] = df['model_full'].map(short_name_map)

    # 设置图表的美学风格
    sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#f7f7f7", "grid.linestyle": "--"})
    plt.figure(figsize=(14, 8))

    # 使用seaborn绘制线图，并传入自定义的颜色调色板
    sns.lineplot(
        data=df,
        x='step',
        y='average_accuracy',
        hue='Model',
        palette=color_palette, # 应用自定义颜色
        marker='o',
        linewidth=2.5,
        markersize=8
    )

    plt.axhline(y=0.5477, color='#1f77b4', linestyle='--', linewidth=2, label='GRPO')

    # 设置图表标题和坐标轴标签
    #plt.title(PLOT_TITLE, fontsize=18, weight='bold')
    plt.xlabel('Training Steps', fontsize=30)
    plt.ylabel('Average Accuracy', fontsize=30)

    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    
    # 确保图例标题正确
    plt.legend(title='Model', fontsize=25, loc='lower right')
    
    plt.grid(True, which='both', linestyle='--', linewidth=1.0)
    plt.tight_layout()

    # 创建输出目录并保存图表
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    plt.savefig(OUTPUT_FILENAME, dpi=300)
    
    print(f"\n图表已成功保存至: '{OUTPUT_FILENAME}'")

# --- 主程序入口 ---
if __name__ == '__main__':
    # 1. 处理所有JSONL文件，生成一个统一的DataFrame
    processed_df = process_evaluation_files(EVAL_OUTPUT_DIR, MODELS_TO_PLOT, BENCHMARKS_TO_AVERAGE)
    
    # 2. 使用生成的DataFrame创建并保存图表
    create_comparison_plot(processed_df, MODELS_TO_PLOT)