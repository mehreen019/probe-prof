#!/usr/bin/env python3
"""
绘制三个模型的response length对比图
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def plot_response_length_comparison(content, smooth=True, window_size=20):
    """
    绘制三个模型的response length对比图
    
    Args:
        content: 要绘制的指标名称
        smooth: 是否使用moving average平滑曲线
        window_size: moving average的窗口大小
    """
    
    # 设置seaborn风格
    sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#f7f7f7", "grid.linestyle": "--"})
    plt.figure(figsize=(14, 8))
    
    # 文件路径
    base_path = os.path.expanduser("~/new_plot/wandb_metrics")
    models = {
        #"GRPO": f"{base_path}/GRPO/{content}.csv",
        #"Blend-PRM-GRPO": f"{base_path}/Blend-PRM-GRPO/{content}.csv", 
        "PROF-GRPO": f"{base_path}/PROF-GRPO/{content}.csv",
        #"PROF-GRPO w/o Separation": f"{base_path}/nsep/{content}.csv",
        "Filter-Nstep": f"{base_path}/lenseg/{content}.csv",
    }
    
    # 颜色和线型
    colors = ["#bf3415", "#1f77b4"] #['#1f77b4', "#d68a27", "#bf3415"]
    line_styles = ['-', '-'] #['-', '-', '-']
    
    # 读取并绘制每个模型的数据
    for i, (model_name, file_path) in enumerate(models.items()):
        try:
            if os.path.exists(file_path):
                # 读取CSV文件
                df = pd.read_csv(file_path)
                
                # 获取指标列名（第一行）
                metric_col = df.columns[0]
                
                # 获取数据（从第二行开始，对应step 1）
                data = df.iloc[1:281, 0].values  # 只取前280步
                
                # 创建training step数组（从1开始）
                steps = np.arange(1, len(data) + 1)
                
                # 应用moving average平滑
                if smooth and len(data) >= window_size:
                    # 使用pandas的rolling mean进行平滑
                    data_smooth = pd.Series(data).rolling(window=window_size, center=True).mean().fillna(method='bfill').fillna(method='ffill')
                    plot_data = data_smooth.values
                    line_label = f"{model_name} (smoothed)"
                else:
                    plot_data = data
                    line_label = model_name
                
                # 绘制线条
                plt.plot(steps, plot_data, 
                        label=line_label, 
                        color=colors[i], 
                        linestyle=line_styles[i],
                        linewidth=2.5,
                        alpha=0.8)
                
                print(f"{model_name}: 成功读取 {len(data)} 步数据" + (" (已平滑)" if smooth else ""))
                
            else:
                print(f"警告: 文件不存在 - {file_path}")
                
        except Exception as e:
            print(f"读取 {model_name} 数据时出错: {e}")
    
    # 设置图形属性
    plt.xlabel('Training Step', fontsize=30)
    plt.ylabel('Number of Reasoning Steps', fontsize=30)
    #plt.title('Response Length Comparison Across Training Steps', fontsize=16, fontweight='bold')

    # 设置x轴和y轴刻度字体大小
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)    
    # 设置坐标轴
    #plt.xlim(1, 280)
    #plt.ylim(0.48, 0.6)
    plt.grid(True, which='both', linestyle=':', linewidth=1)
    
    # 添加图例
    plt.legend(fontsize=25, loc='upper left')
    
    # 优化布局
    plt.tight_layout()
    
    # 保存图片
    output_file = os.path.expanduser(f"~/new_plot/plots/{content}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n图片已保存为: {output_file}")
    
    # 显示图片
    plt.show()


if __name__ == "__main__":
    print("=== Response Length 对比图绘制工具 ===")
    print("正在读取三个模型的数据...")
    content = 'train_len_segs_mean' # 'actor_entropy_loss'
    
    # 绘制平滑后的对比图（默认）
    print("绘制平滑后的对比图...")
    plot_response_length_comparison(content, smooth=False, window_size=20)
