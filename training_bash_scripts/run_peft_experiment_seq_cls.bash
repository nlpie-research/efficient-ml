#!/usr/bin/env bash
# all tasks 50, 200, 500 samples - put tasks you wanna do in the tasks array separated by space
# nlpie/tiny-biobert nlpie/bio-mobilebert 
                    
# model_name_or_path=(nlpie/tiny-biobert
#                     nlpie/distil-biobert 
#                     roberta-base
#                     emilyalsentzer/Bio_ClinicalBERT                                                                                                 
#                      )
# peft_methods=(Full) # LORA PREFIX PROMPT
model_name_or_path=(roberta-base)
peft_methods=(LORA)
task=icd9-triage-no-category-in-text
max_epochs=5
gpu=3
# data_dir=/mnt/sdd/efficient_ml_data/datasets/icd9-triage/no_category_in_text
# eval_data_dir=/mnt/sdd/efficient_ml_data/datasets/icd9-triage/no_category_in_text
log_save_dir=/mnt/sdd/efficient_ml_data/saved_models/peft/logs
ckpt_save_dir=/mnt/sdd/efficient_ml_data/saved_models/peft/ckpts
for model in "${model_name_or_path[@]}"
    do
    for peft_method in "${peft_methods[@]}"
        do
        CUDA_VISIBLE_DEVICES="$gpu" 
        python peft_trainer.py \
            --model_name_or_path "$model" \
            --max_epochs "$max_epochs" \
            --task "$task" \
            --peft_method "$peft_method" \
            --log_save_dir $log_save_dir \
            --ckpt_save_dir $ckpt_save_dir
    done
done