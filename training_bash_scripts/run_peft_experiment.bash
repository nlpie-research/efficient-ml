#!/usr/bin/env bash
model_name_or_path=(nlpie/distil-biobert)
peft_methods=(LORA)
task=i2b2-2012-NER
max_epochs=5
gpu=3
log_save_dir=/mnt/sdd/efficient_ml_data/saved_models/peft/logs
ckpt_save_dir=/mnt/sdd/efficient_ml_data/saved_models/peft/ckpts
for model in "${model_name_or_path[@]}"
    do
    for peft_method in "${peft_methods[@]}"
        do
        python peft_trainer.py \
            --model_name_or_path $model \
            --max_epochs $max_epochs \
            --task $task \
            --peft_method $peft_method \
            --log_save_dir $log_save_dir \
            --ckpt_save_dir $ckpt_save_dir
    done
done