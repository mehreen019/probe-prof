set -x

export RAY_TMPDIR=/opt/dlami/nvme/ray_tmp
export VLLM_ATTENTION_BACKEND=XFORMERS
data=numina_math
project_name=qwen_math_1-5b_prm_filter
algorithm=grpo
model=Qwen2.5-Math-1.5B
model_name_or_path=Qwen/$model
n=8
experiment_name=${model}-${algorithm}-${data}-n${n}
GPUS=(0 1 2 3)
my_world_size=${#GPUS[@]}

math_train_path=$HOME/PRM_filter/data/$data/normal_train.parquet
math_test_path=$HOME/PRM_filter/data/math500/normal_test.parquet 

train_files="['$math_train_path']"
test_files="['$math_test_path']"

mkdir -p logs/${project_name}

CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}") python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$algorithm \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$model_name_or_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=$my_world_size \
    trainer.val_before_train=True \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.default_local_dir=/opt/dlami/nvme/${USER}_ckpoints/${project_name}/${experiment_name} \
    trainer.test_freq=20 \
    trainer.total_epochs=10 \
    reward_model.reward_manager='prof' \
    2>&1 | tee logs/${project_name}/${experiment_name}.log