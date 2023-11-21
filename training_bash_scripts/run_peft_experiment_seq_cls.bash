#!/usr/bin/env bash
# all tasks 50, 200, 500 samples - put tasks you wanna do in the tasks array separated by space
# nlpie/tiny-biobert nlpie/bio-mobilebert 
                    
# model_name_or_path=(nlpie/tiny-biobert
#                     nlpie/distil-biobert 
#                     roberta-base
#                     emilyalsentzer/Bio_ClinicalBERT
# /mnt/sdc/niallt/saved_models/declutr/mimic/few_epoch/mimic-roberta-base/2_anch_2_pos_min_1024/transformer_format/                                                                                                 
#                      )
# peft_methods=(Full) # LORA PREFIX_TUNING PROMPT_TUNING P_TUNING
# model_name_or_path=(nlpie/bio-mobilebert
#                     nlpie/tiny-biobert
#                     roberta-base
#                     nlpie/distil-biobert
#                     dmis-lab/biobert-v1.1)
model_name_or_path=(nlpie/tiny-biobert)
peft_methods=(LORA)
tasks=(mimic-mp)
max_epochs=5
gpu=0
log_save_dir=/mnt/sdd/efficient_ml_data/saved_models/peft/Optuna/logs
ckpt_save_dir=/mnt/sdd/efficient_ml_data/saved_models/peft/Optuna/ckpts
for task in "${tasks[@]}"
    do
    for model in "${model_name_or_path[@]}"
        do
        for peft_method in "${peft_methods[@]}"
            do
            export CUDA_VISIBLE_DEVICES="$gpu"
            python peft_trainer.py \
                --model_name_or_path "$model" \
                --max_epochs "$max_epochs" \
                --task "$task" \
                --peft_method "$peft_method" \
                --log_save_dir $log_save_dir \
                --ckpt_save_dir $ckpt_save_dir \
                --few_shot_n=256
        done
    done
done