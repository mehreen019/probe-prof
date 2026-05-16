#!/usr/bin/env python3
"""
简化版数学性能对比图绘制工具
使用seaborn绘制GRPO和PROF-SEP在不同数据集上的表现对比
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_math_performance():
    """
    绘制数学性能对比图，类似参考图片的样式
    """
    # 设置seaborn风格
    sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#f7f7f7", "grid.linestyle": "--"})
    
    # 创建图形
    plt.figure(figsize=(16, 10))
    
    # 数据集名称（按照你的数据顺序）
    datasets = ['MATH-500', 'Minerva Math', 'OlympiadBench', 'AMC2023', 'AIME2024']
    
    # 直接使用你提供的平均值数据
    grpo_means = [0.6785820084211174, 0.25760753282587145, 0.3723327258704411, 0.5417358637242945, 0.15117452426046174]
    prof_sep_means = [0.7703847306581121, 0.6314211209673033, 0.5309357955286576, 0.6338939725178833, 0.26232013012904115]
    
    # 转换为百分比
    grpo_means_pct = [score * 100 for score in grpo_means]
    prof_sep_means_pct = [score * 100 for score in prof_sep_means]
    
    # 设置x轴位置
    x = np.arange(len(datasets))
    width = 0.35
    
    # 绘制柱状图
    bars1 = plt.bar(x - width/2, prof_sep_means_pct, width, label='PROF-GRPO', color='#ff7f0e', alpha=0.8)
    bars2 = plt.bar(x + width/2, grpo_means_pct, width, label='GRPO', color='#1f77b4', alpha=0.8)

    
    # 设置图形属性
    #plt.xlabel('Dataset', fontsize=20)
    plt.ylabel('MC Estimation Accuracy (%)', fontsize=25)
    #plt.title('Mathematical Performance Comparison: GRPO vs PROF-SEP', fontsize=22, fontweight='bold')
    plt.xticks(x, datasets, ha='center', fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(fontsize=25, loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # 设置y轴范围
    plt.ylim(0, 85)
    
    # 在柱状图上添加数值标签
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=25)
    
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=25)
    
    # 优化布局
    plt.tight_layout()
    
    # 保存图片
    output_file = "./plots/mc_prm_bar.png"
    plt.savefig(output_file, bbox_inches='tight')
    print(f"\n图片已保存为: {output_file}")
    
    # 显示图片
    plt.show()
    
    # 打印数据对比
    print("\n" + "="*60)
    print("性能对比总结:")
    print("="*60)
    for i, dataset in enumerate(datasets):
        grpo_pct = grpo_means_pct[i]
        prof_sep_pct = prof_sep_means_pct[i]
        improvement = prof_sep_pct - grpo_pct
        print(f"{dataset}: GRPO {grpo_pct:.1f}% → PROF-SEP {prof_sep_pct:.1f}% (提升 {improvement:+.1f}%)")
    
    # 计算平均提升
    avg_improvement = np.mean([prof_sep_means_pct[i] - grpo_means_pct[i] for i in range(len(datasets))])
    print(f"\n平均提升: {avg_improvement:+.1f}%")

if __name__ == "__main__":
    print("=== 数学性能对比图绘制工具 ===")
    print("注意：请先替换脚本中的示例数据为你的实际数据！")
    
    # 绘制对比图
    plot_math_performance()
