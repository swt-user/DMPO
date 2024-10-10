model_name=Mistral-7B-Instruct-v0.2
task=$1

exp_name=mistral_dmpo
gpu_num=4  
num_workers=4
model_path=$2
save_dir=$3 

sft_data_path="data/${task}_sft.json"
batch_size=64
micro_batch_size=4
accumulation_step=$((${batch_size}/${gpu_num}/${micro_batch_size}))

sft_model_name=${exp_name}-${model_name}-${task}-sft

python -u -m fastchat.serve.controller >> logs/${exp_name}-controller.log 2>&1 &
fs_controller_pid=$!
pm_data_path=data_pm/${task}_pm_${exp_name}.json


# Part 3: DMPO model training
batch_size=32
micro_batch_size=2
accumulation_step=$((${batch_size}/${gpu_num}/${micro_batch_size}))
beta=0.1
lr=1e-6

dpo_model_name=${exp_name}-${model_name}-${task}-dpo

python -m torch.distributed.run --nproc_per_node=${gpu_num} --master_port=20001 fastchat/train/train_dpo_mistral.py \
    --model_name_or_path ${save_dir}${sft_model_name} \
    --ref_model_name_or_path ${save_dir}${sft_model_name} \
    --data_path ${pm_data_path} \
    --bf16 True \
    --output_dir ${save_dir}${dpo_model_name} \
    --num_train_epochs 3 \
    --per_device_train_batch_size ${micro_batch_size} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${accumulation_step} \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_total_limit 5 \
    --beta ${beta} \
    --learning_rate ${lr} \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "constant_with_warmup" \
    --logging_steps 5 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 4096 \
    --max_prompt_length 512 \
    --max_target_length 3072 \
    --gradient_checkpointing True \
    --lazy_preprocess False

if [ $? -ne 0 ]; then
    echo "Preference model training failed"
    exit 1
fi

fs_worker_port=21002
CUDA_VISIBLE_DEVICES=0 python -u -m fastchat.serve.model_worker --model-path ${save_dir}${dpo_model_name} --port ${fs_worker_port} --worker-address http://localhost:${fs_worker_port} >> logs/model_worker.log 2>&1 &

fs_worker_pid=$!
sleep 60

python -m eval_agent.main --agent_config fastchat --model_name ${dpo_model_name} --exp_config ${task} --split test

if [ $? -ne 0 ]; then
    echo "base agent evaluation failed"
    kill -9 $fs_worker_pid
    exit 1
fi

kill -9 $fs_worker_pid

kill -9 $fs_controller_pid
