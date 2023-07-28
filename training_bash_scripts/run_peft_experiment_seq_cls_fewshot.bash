#!/usr/bin/env bash
# all tasks 50, 200, 500 samples - put tasks you wanna do in the tasks array separated by space
# nlpie/tiny-biobert nlpie/bio-mobilebert 
                    
# model_name_or_path=(nlpie/tiny-biobert
#                     nlpie/distil-biobert 
#                     roberta-base
#                     emilyalsentzer/Bio_ClinicalBERT                                                                                                 
#                      )
# peft_methods=(Full) # LORA PREFIX PROMPT
# nlpie/bio-mobilebert
model_name_or_path=(nlpie/tiny-biobert
                    roberta-base
                    nlpie/distil-biobert
                    emilyalsentzer/Bio_ClinicalBERT
                    /mnt/sdc/niallt/saved_models/declutr/mimic/few_epoch/mimic-roberta-base/2_anch_2_pos_min_1024/transformer_format/)
peft_methods=(LORA)
tasks=(i2b2-2010-RE mimic-los mimic-mp)
max_epochs=5
few_shot_num=(16 32 64 128 200)
gpu=3
log_save_dir=/mnt/sdd/efficient_ml_data/saved_models/peft/logs
ckpt_save_dir=/mnt/sdd/efficient_ml_data/saved_models/peft/ckpts
for task in "${tasks[@]}"
    do
    for model in "${model_name_or_path[@]}"
        do
        for num in "${few_shot_num[@]}"
            do
            for peft_method in "${peft_methods[@]}"
                do
                CUDA_VISIBLE_DEVICES="$gpu" python peft_trainer.py \
                    --model_name_or_path "$model" \
                    --max_epochs "$max_epochs" \
                    --task "$task" \
                    --peft_method "$peft_method" \
                    --log_save_dir $log_save_dir \
                    --ckpt_save_dir $ckpt_save_dir \
                    --few_shot_n "$num"
            done
        done
    done
done

