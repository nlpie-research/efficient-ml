#!/usr/bin/env bash
# add sentence-transformers/all-mpnet-base-v2 nlpie/distil-biobert nlpie/bio-mobilebert nlpie/tiny-biobert                    
model_name_or_path=(roberta-base
                    nlpie/distil-biobert
                    nlpie/tiny-biobert
                    )
peft_methods=(PROMPT_TUNING) # PREFIX_TUNING LORA PROMPT_TUNING
tasks=(i2b2-2010-NER i2b2-2012-NER i2b2-2014-NER) # i2b2-2010-NER 
max_epochs=5
gpu=4
log_save_dir=/mnt/sdd/efficient_ml_data/saved_models/peft/logs
ckpt_save_dir=/mnt/sdd/efficient_ml_data/saved_models/peft/ckpts
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
                --eval_batch_size 32
        done
    done
done