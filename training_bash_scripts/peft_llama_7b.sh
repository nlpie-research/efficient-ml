#!/usr/bin/env bash
# decapoda-research/llama-7b-hf ybelkada/falcon-7b-sharded-bf16 tiiuae/falcon-7b-instruct
model_name_or_path=(ybelkada/falcon-7b-sharded-bf16                                                                 
                     )
task=icd9-triage-no-category-in-text
num_sample=(64)
peft_methods=(LORA)
max_epochs=5
gpu=3
data_dir=/mnt/sdc/niallt/mimic_iii/processed/HADM_ID_split/icd9-triage/no_category_in_text/
eval_data_dir=/mnt/sdc/niallt/mimic_iii/processed/HADM_ID_split/icd9-triage/no_category_in_text/
for model in "${model_name_or_path[@]}"
    do
    for peft_method in "${peft_methods[@]}"
        do
        for num in "${num_sample[@]}"
            do
            CUDA_VISIBLE_DEVICES="$gpu" python peft_trainer_v2.py --model_name_or_path "$model" \
                                    --max_epochs "$max_epochs" \
                                    --training_data_dir "$data_dir" \
                                    --eval_data_dir "$eval_data_dir" \
                                    --task "$task" \
                                    --few_shot_n "$num" \
                                    --peft_method "$peft_method" \
                                    --training_size fewshot \
                                    --train_batch_size 2 \
                                    --eval_batch_size 2 \
                                    --eight_bit_training
        done
    done
done