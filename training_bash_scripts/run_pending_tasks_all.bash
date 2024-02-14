#!/usr/bin/env bash
# target_model=$1
# gpu=$2
gpu=1
max_epochs=5
log_save_dir=/mnt/sdd/efficient_ml_data/saved_models/peft/logs
ckpt_save_dir=/mnt/sdd/efficient_ml_data/saved_models/peft/ckpts
pending_tasks_file=./training_bash_scripts/all_pending_tasks.csv
for pt in $(cat "$pending_tasks_file")
do
    IFS=$',' read -r model_name_or_path task peft_method <<< "$pt"
    
    # Check if the model is the target model
    # if [ "$model_name_or_path" != "$target_model" ]; then
    #     continue
    # fi
    
    echo $model_name_or_path
    echo $task
    echo $peft_method
    
    export CUDA_VISIBLE_DEVICES="$gpu"
    python peft_trainer.py \
        --model_name_or_path "$model_name_or_path" \
        --max_epochs "$max_epochs" \
        --task "$task" \
        --peft_method "$peft_method" \
        --log_save_dir $log_save_dir \
        --ckpt_save_dir $ckpt_save_dir \
        --saving_strategy epoch
done