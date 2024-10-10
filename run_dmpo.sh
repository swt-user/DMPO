model_name=Llama-2-7b-chat-hf
task=$1

exp_name=llama-dmpo
gpu_num=4  
num_workers=4
model_path=$2
save_dir=$3 

# Part 1: SFT training & evaluation
sft_data_path="data/${task}_sft.json"
batch_size=64
micro_batch_size=4
accumulation_step=$((${batch_size}/${gpu_num}/${micro_batch_size}))

sft_model_name=${exp_name}-${model_name}-${task}-sft


python -m torch.distributed.run --nproc_per_node=${gpu_num} --master_port=20001 fastchat/train/train.py \
    --model_name_or_path ${model_path}${model_name} \
    --data_path ${sft_data_path} \
    --bf16 True \
    --output_dir ${save_dir}${sft_model_name} \
    --num_train_epochs 3 \
    --per_device_train_batch_size ${micro_batch_size} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${accumulation_step} \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --lazy_preprocess False

if [ $? -ne 0 ]; then
    echo "SFT training failed"
    exit 1
fi

python -u -m fastchat.serve.controller >> logs/${exp_name}-controller.log 2>&1 &
fs_controller_pid=$!

fs_worker_port=21002
CUDA_VISIBLE_DEVICES=0 python -u -m fastchat.serve.model_worker --model-path ${save_dir}${sft_model_name} --port ${fs_worker_port} --worker-address http://localhost:${fs_worker_port} >> logs/${exp_name}-model_worker.log 2>&1 &

fs_worker_pid=$!
sleep 60

# evaluation
python -m eval_agent.main --agent_config fastchat --model_name ${sft_model_name} --exp_config ${task} --split test

if [ $? -ne 0 ]; then
    echo "base agent evaluation failed"
    kill -9 $fs_worker_pid
    exit 1
fi

kill -9 $fs_worker_pid




# Part 2: build dataset for dpo training
explore_model_name=${sft_model_name}-explore

for ((j=0;j<${num_workers};j=j+1)); do
    if [ -d "${save_dir}${explore_model_name}-${j}" ]; then
        echo "Link to model exists"
    else
        ln -s ${save_dir}${sft_model_name} ${save_dir}${explore_model_name}-${j}
    fi
done
if [ -f "logs/${exp_name}-worker_pid.txt" ]; then
    rm logs/${exp_name}-worker_pid.txt
fi

fs_worker_port=21002
worker_idx=0
for ((j=0;j<${num_workers};j=j+1)); do
    echo "Launch the model worker on port ${fs_worker_port}"
    CUDA_VISIBLE_DEVICES=$((${worker_idx} % ${gpu_num})) python -u -m fastchat.serve.model_worker \
        --model-path ${save_dir}${explore_model_name}-${j} \
        --port ${fs_worker_port} \
        --worker-address http://localhost:${fs_worker_port} >> logs/${exp_name}-model_worker-${j}.log 2>&1 &
    echo $! >> logs/${exp_name}-worker_pid.txt
    fs_worker_port=$(($fs_worker_port+1))
    worker_idx=$(($worker_idx+1))
    sleep 15
done

sleep 60

echo "Base agent starts exploring"
if [ -f "logs/${exp_name}-eval_pid.txt" ]; then
    rm logs/${exp_name}-eval_pid.txt
fi
for ((j=0;j<${num_workers};j=j+1)); do
    python -m eval_agent.main --agent_config fastchat --model_name ${explore_model_name}-${j} --exp_config ${task} --split train --part_num ${num_workers} --part_idx ${j} &
    echo $! >> logs/${exp_name}-eval_pid.txt
done

wait $(cat logs/${exp_name}-eval_pid.txt)
rm logs/${exp_name}-eval_pid.txt
echo "Training data collection finished"

if [ $? -ne 0 ]; then
    echo "Training data collection failed"
    kill -9 $(cat logs/${exp_name}-worker_pid.txt)
    rm logs/${exp_name}-worker_pid.txt
    exit 1
fi

echo "Kill the model workers"
kill -9 $(cat logs/${exp_name}-worker_pid.txt)
rm logs/${exp_name}-worker_pid.txt

echo "Build preference data"
pm_data_path=data_pm/${task}_pm_${exp_name}.json
python construct_preference_nocut.py --model ${explore_model_name} --task $task --golden_traj_path $sft_data_path --output_path $pm_data_path





# Part 3: DMPO model training
batch_size=32
micro_batch_size=2
accumulation_step=$((${batch_size}/${gpu_num}/${micro_batch_size}))
beta=0.1
lr=1e-6

dpo_model_name=${exp_name}-${model_name}-${task}-dmpo

# ETO: train_dpo.py / train_dpo_mistral.py
python -m torch.distributed.run --nproc_per_node=${gpu_num} --master_port=20001 fastchat/train/train_dmpo_efficient.py \
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
