#!/usr/bin/env bash
# all tasks 50, 200, 500 samples - put tasks you wanna do in the tasks array separated by space
# model_name_or_path=(roberta-base
#                     michiyasunaga/LinkBERT-base
#                     emilyalsentzer/Bio_ClinicalBERT
#                     /mnt/sdc/niallt/saved_models/language_modelling/mimic/mimic-roberta-base/sampled_250000/22-12-2022--12-45/checkpoint-100000/
#                     /mnt/sdc/niallt/saved_models/declutr/mimic/few_epoch/mimic-roberta-base/2_anch_2_pos_min_1024/transformer_format/                                                           
#                      )
# model_name_or_path=(nlpie/distil-biobert 
#                     /mnt/sdc/niallt/saved_models/language_modelling/mimic/mimic-roberta-base/sampled_250000/22-12-2022--12-45/checkpoint-100000/                                                                             
#                      )
# num_sample=(16 32 64 200)
# peft_methods=(LORA PREFIX_TUNING PROMPT_TUNING P_TUNING)
# model_name_or_path=(nlpie/distil-biobert)
# num_sample=(200)
# peft_methods=(LORA)
# task=icd9-triage-no-category-in-text
# max_epochs=5
# gpu=3
# data_dir=/mnt/sdc/niallt/mimic_iii/processed/HADM_ID_split/icd9-triage/no_category_in_text
# eval_data_dir=/mnt/sdc/niallt/mimic_iii/processed/HADM_ID_split/icd9-triage/no_category_in_text
# log_save_dir=/mnt/sdd/efficient_ml_data/saved_models/peft/logs
# ckpt_save_dir=/mnt/sdd/efficient_ml_data/saved_models/peft/ckpts
# for model in "${model_name_or_path[@]}"
#     do
#     for peft_method in "${peft_methods[@]}"
#         do
#         for num in "${num_sample[@]}"
#             do
#             # CUDA_VISIBLE_DEVICES="$gpu" 
#             python peft_trainer.py \
#                 --model_name_or_path "$model" \
#                 --max_epochs "$max_epochs" \
#                 --training_data_dir "$data_dir" \
#                 --eval_data_dir "$eval_data_dir" \
#                 --task "$task" \
#                 --few_shot_n "$num" \
#                 --peft_method "$peft_method" \
#                 --training_size fewshot \
#                 --log_save_dir $log_save_dir \
#                 --ckpt_save_dir $ckpt_save_dir
#         done
#     done
# done

model_name_or_path=(roberta-base)
peft_methods=(LORA)
task=i2b2-2012-NER
max_epochs=5
gpu=3
data_dir=/mnt/sdd/niallt/bio-lm/data/tasks/i2b2-2012_hf_dataset/
log_save_dir=/mnt/sdd/efficient_ml_data/saved_models/peft/logs
ckpt_save_dir=/mnt/sdd/efficient_ml_data/saved_models/peft/ckpts
for model in "${model_name_or_path[@]}"
    do
    for peft_method in "${peft_methods[@]}"
        do
        python peft_trainer.py \
            --model_name_or_path $model \
            --max_epochs $max_epochs \
            --data_dir $data_dir \
            --task $task \
            --task_type TOKEN_CLS \
            --peft_method $peft_method \
            --log_save_dir $log_save_dir \
            --ckpt_save_dir $ckpt_save_dir
    done
done