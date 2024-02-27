#!/usr/bin/env bash
# add sentence-transformers/all-mpnet-base-v2 nlpie/distil-biobert nlpie/bio-mobilebert nlpie/tiny-biobert                    
# model_name_or_path=(johngiorgi/declutr-base
#                     /mnt/sdc/niallt/saved_models/declutr/mimic/few_epoch/roberta-base/2_anch_2_pos_min_1024/transformer_format/
#                     roberta-base
#                     /mnt/sdc/niallt/saved_models/language_modelling/mimic/mimic-roberta-base/sampled_250000/22-12-2022--12-45/checkpoint-100000/
#                     /mnt/sdc/niallt/saved_models/language_modelling/mimic/roberta-base-mimic-note-custom_pretraining_max_epoch_2_weighted/sampled_250000/07-07-2023--08-30/checkpoint-30000/
#                     /mnt/sdc/niallt/saved_models/declutr/mimic/few_epoch/mimic-roberta-base/2_anch_2_pos_min_1024/transformer_format/
#                     )
model_name_or_path=(nlpie/distil-biobert)
peft_methods=(LORA) # PREFIX_TUNING LORA PROMPT_TUNING
tasks=(i2b2-2014-NER) # i2b2-2010-NER | i2b2-2010-NER i2b2-2012-NER 
max_epochs=1
gpu=0
log_save_dir=/mnt/sdd/efficient_ml_data/saved_models/peft/debugging/logs
ckpt_save_dir=/mnt/sdd/efficient_ml_data/saved_models/peft/debugging/ckpts
for task in "${tasks[@]}"
    do
    for model in "${model_name_or_path[@]}"
        do
        for peft_method in "${peft_methods[@]}"
            do
            CUDA_VISIBLE_DEVICES="$gpu" python peft_trainer.py \
                --model_name_or_path $model \
                --max_epochs $max_epochs \
                --task $task \
                --peft_method $peft_method \
                --log_save_dir $log_save_dir \
                --ckpt_save_dir $ckpt_save_dir \
                --train_batch_size 32 \
                --eval_batch_size 32 \
                --saving_strategy epoch
        done
    done
done