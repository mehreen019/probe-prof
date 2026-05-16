#!/usr/bin/env python3
"""
简单的wandb指标导出工具
专门用于导出指定项目的特定指标到CSV
"""

import wandb
import pandas as pd
import argparse
import os

def export_specific_metrics(run_path, metrics, output_file=None):
    """
    导出指定运行的特定指标到CSV
    
    Args:
        run_path: 运行路径，例如 "machine-learning0/qwen_math_7b_prm_filter/gdklojdp"
        metrics: 要导出的指标列表
        output_file: 输出文件名，如果为None则自动生成
    """
    try:
        print(f"正在连接wandb运行: {run_path}")
        
        # 初始化wandb API
        api = wandb.Api()
        run = api.run(run_path)
        
        print(f"成功连接到运行: {run.name}")
        
        # 获取运行历史数据
        history = run.history()
        
        if history.empty:
            print("警告: 该运行没有历史数据")
            return
        
        print(f"找到 {len(history)} 行数据")
        print(f"可用列: {list(history.columns)}")
        
        # 检查哪些指标存在
        available_metrics = []
        missing_metrics = []
        
        for metric in metrics:
            if metric in history.columns:
                available_metrics.append(metric)
            else:
                missing_metrics.append(metric)
        
        if available_metrics:
            print(f"\n将导出以下指标: {available_metrics}")
            # 只保留需要的指标列
            filtered_data = history[available_metrics]

            for col in filtered_data.columns:
                filtered_data = filtered_data[filtered_data[col] != '']
            
            for metric in available_metrics:
                # 生成输出文件名
                metric_name = metric.replace("/", "_")
                final_output_file = f"{output_file}/{metric_name}.csv"
            
                # 保存到CSV
                filtered_data[metric].to_csv(final_output_file, index=False)
                print(f"\n成功导出到: {final_output_file}")
            
                # 显示数据预览
                print("\n数据预览:")
                print(filtered_data[metric].head())
            
        if missing_metrics:
            print(f"\n以下指标未找到: {missing_metrics}")
            print("请检查指标名称是否正确")
            
    except Exception as e:
        print(f"导出过程中出错: {e}")

def main():
    parser = argparse.ArgumentParser(description="从wandb导出指定指标到CSV")
    parser.add_argument("run_path", help="运行路径 (格式: username/project/run_id)")
    parser.add_argument("--metrics", nargs="+", 
                       default=["critic/score_before_filter", "actor/entropy_loss", "response_length/mean"],
                       help="要导出的指标列表")
    parser.add_argument("--output", help="输出文件名")
    
    args = parser.parse_args()
    
    print("=== Wandb指标导出工具 ===")
    print(f"运行路径: {args.run_path}")
    print(f"要导出的指标: {args.metrics}")
    print()
    
    export_specific_metrics(args.run_path, args.metrics, args.output)

if __name__ == "__main__":
    main()
