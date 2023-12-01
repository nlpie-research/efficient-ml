#!/usr/bin/env bash
target_task=$1
gpu=$2
max_epochs=5
log_save_dir=/mnt/sdd/efficient_ml_data/saved_models/peft/logs
ckpt_save_dir=/mnt/sdd/efficient_ml_data/saved_models/peft/ckpts
pending_tasks_file=./training_bash_scripts/all_pending_tasks.csv
for pt in $(cat "$pending_tasks_file")
do
    IFS=$',' read -r model_name_or_path task peft_method <<< "$pt"
    
    # Check if the task is the target task
    if [ "$task" != "$target_task" ]; then
        continue
    fi
    
    echo $model_name_or_path
    echo $task
    echo $peft_method
    echo
    
    export CUDA_VISIBLE_DEVICES="$gpu"
    python peft_trainer.py \
        --model_name_or_path "$model_name_or_path" \
        --max_epochs "$max_epochs" \
        --task "$task" \
        --peft_method "$peft_method" \
        --log_save_dir $log_save_dir \
        --ckpt_save_dir $ckpt_save_dir
done