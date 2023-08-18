#!/usr/bin/env bash
# method=$1
max_epochs=5
gpu=0
log_save_dir=/mnt/sdd/efficient_ml_data/saved_models/peft/logs
ckpt_save_dir=/mnt/sdd/efficient_ml_data/saved_models/peft/ckpts
pending_tasks_file=./training_bash_scripts/pending_tasks_SEQ_CLS.csv
few_shot_num=(16 32 64 128 200)
for pt in $(cat "$pending_tasks_file")
do
    IFS=$',' read -r model_name_or_path task peft_method <<< "$pt"
    
    # if [ "$peft_method" == "$method" ]; then
    echo $model_name_or_path
    echo $task
    echo $peft_method
    split=(${model_name_or_path//\// })
    model_name=${split[-1]}
    run_logs="./Runs/"$model_name"_"$task"_"$peft_method".log"
    echo
    
    # for num in "${few_shot_num[@]}"
    # do            
    export CUDA_VISIBLE_DEVICES="$gpu"
    python peft_trainer.py \
        --model_name_or_path "$model_name_or_path" \
        --max_epochs "$max_epochs" \
        --task "$task" \
        --peft_method "$peft_method" \
        --log_save_dir $log_save_dir \
        --ckpt_save_dir $ckpt_save_dir > "$run_logs"
    # fi
done