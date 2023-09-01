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
#                   nlpie/tiny-biobert
                    # roberta-base
                    # nlpie/distil-biobert
                    # emilyalsentzer/Bio_ClinicalBERT
                    # /mnt/sdc/niallt/saved_models/declutr/mimic/few_epoch/mimic-roberta-base/2_anch_2_pos_min_1024/transformer_format/
                    # /mnt/sdc/niallt/saved_models/language_modelling/mimic/roberta-base-mimic-note-custom_pretraining_contrastive_max_epoch_2_weighted/sampled_250000/22-08-2023--16-32/checkpoint-14000/
                    # /mnt/sdc/niallt/saved_models/language_modelling/mimic/roberta-base-mimic-note-custom_pretraining_max_epoch_2_weighted/sampled_250000/07-07-2023--08-30/checkpoint-30000/                    
model_name_or_path=(/mnt/sdc/niallt/saved_models/language_modelling/mimic/roberta-base-mimic-note-custom_pretraining_contrastive_max_epoch_2_weighted/sampled_250000/23-08-2023--08-42/checkpoint-14000)
peft_methods=(LORA)
tasks=(ICD9-Triage) # i2b2-2010-RE mimic-los mimic-mp ICD9-Triage
max_epochs=5
few_shot_num=(128)
gpu=0
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

