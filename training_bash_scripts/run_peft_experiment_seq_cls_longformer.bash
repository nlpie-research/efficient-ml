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
model_name_or_path=(yikuan8/Clinical-Longformer)
peft_methods=(LORA)
tasks=(icd9-triage)
max_epochs=5
gpu=1
log_save_dir=/mnt/sdd/efficient_ml_data/saved_models/peft/logs
ckpt_save_dir=/mnt/sdd/efficient_ml_data/saved_models/peft/ckpts
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
                --train_batch_size 2 \
                --eval_batch_size 2 \
                --max_length 4096
        done
    done
done