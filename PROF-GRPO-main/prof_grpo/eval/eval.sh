temperature=1.0
my_world_size=1
K=16
PROMPT_TYPE="qwen25-math-cot"
model_name="{YOUR_MODEL_NAME}"
output_dir="{YOUR_CKPT_DIR}/${model_name}_temp${temperature}_K${K}"
text_file_dir="{YOUR_TEXT_FILE_DIR}"
DATA_NAME="math500,minerva_math,olympiadbench,aime24,amc23"

mkdir -p ${output_dir}
mkdir -p ${text_file_dir}

for i in $(seq 20 20 280); do
    CUDA_VISIBLE_DEVICES=0 python main_eval.py \
        --model_name_or_path {YOUR_CKPT_DIR}/${model_name}/global_step_$i/ \
        --data_names ${DATA_NAME} \
        --output_dir ${output_dir}/global_step_$i \
        --prompt_type ${PROMPT_TYPE} \
        --seed 0 \
        --temperature ${temperature} \
        --K ${K} \
        --my_world_size ${my_world_size} \
        --text_file_dir ${text_file_dir} \
        --local_index 0 \
        --step $i \
        --model_name ${model_name}
done
        
