output_dir="$HOME/new_plot/wandb_metrics/lenseg"
mkdir -p $output_dir

python wandb_export.py machine-learning0/qwen_math_7b_prm_filter/6gs1kmqb \
    --output $output_dir \
    --metrics 'train/len_segs_mean'