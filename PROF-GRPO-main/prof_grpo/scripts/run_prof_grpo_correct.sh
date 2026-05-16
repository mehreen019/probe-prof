set -x

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAY_TMPDIR=/opt/dlami/nvme/ray_tmp
export VLLM_ATTENTION_BACKEND=XFORMERS
data=numina_math
project_name=qwen_math_7b_prm_filter
algorithm=grpo
model=Qwen2.5-Math-7B
model_name_or_path=Qwen/$model
prm_model_path=Qwen/Qwen2.5-Math-PRM-7B # grm: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B; verify: Qwen/Qwen2.5-Math-PRM-7B

# for test: change n=4, train_prompt_bsz=64, train_prompt_mini_bsz=64, GPUS=2
# for training: change n=8, train_prompt_bsz=512, train_prompt_mini_bsz=512, GPUS=4

enable_filter_groups=True
enable_prm=True # true if use prm filter
prof_filter=4 # keep 8 trajectories out of n 
len_step_reward_coef=10.0
gamma=0.99
n=8
clip_ratio_high=0.2
max_step=30


experiment_name=${model}-${algorithm}-${data}-prm${prof_filter}-n${n}-Correct-Mean-clip${clip_ratio_high}
GPUS=(0 1 2 3 4 5 6 7)
my_world_size=${#GPUS[@]}

math_train_path=$HOME/PRM_filter/data/$data/normal_train.parquet
math_test_path=$HOME/PRM_filter/data/math500/normal_test.parquet 

train_files="['$math_train_path']"
test_files="['$math_test_path']"

mkdir -p logs/${project_name}

CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}") python3 -m prof_grpo_correct.main_prof --config-name prof_trainer \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$model_name_or_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.max_step=${max_step} \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.prof_filter=${prof_filter} \
    algorithm.filter_groups.len_step_reward_coef=${len_step_reward_coef} \
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
    trainer.total_epochs=1 \
    reward_model.micro_batch_size_per_gpu=32 \
    reward_model.model.enable=${enable_prm} \
    reward_model.model.gamma=${gamma} \
    reward_model.model.path=${prm_model_path} \
    reward_model.model.use_remove_padding=True \
    reward_model.model.use_fused_kernels=False \
    reward_model.reward_manager='prof' \
    2>&1 | tee logs/${project_name}/${experiment_name}.log