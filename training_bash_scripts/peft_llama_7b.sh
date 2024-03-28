#!/usr/bin/env bash
# decapoda-research/llama-7b-hf ybelkada/falcon-7b-sharded-bf16 tiiuae/falcon-7b-instruct
model_name_or_path=(
    meta-llama/Llama-2-7b-hf                                                                
                     )
task=mimic-mp
num_sample=(4096)
peft_methods=(Frozen_PLM)
max_epochs=5
gpu=1
log_save_dir=/mnt/sdh/effecient_ml/fewshot_budget/logs
ckpt_save_dir=/mnt/sdh/effecient_ml/fewshot_budget/ckpts
for model in "${model_name_or_path[@]}"
    do
    for peft_method in "${peft_methods[@]}"
        do
        for num in "${num_sample[@]}"
            do
            CUDA_VISIBLE_DEVICES="$gpu" python peft_trainer.py --model_name_or_path "$model" \
                                    --max_epochs "$max_epochs" \
                                    --task "$task" \
                                    --few_shot_n "$num" \
                                    --peft_method "$peft_method" \
                                    --train_batch_size 1 \
                                    --eval_batch_size 1 \
                                    --learning_rate 0.00005 \
                                    --lora_rank 16 \
                                    --lora_alpha 32 \
                                    --gradient_accumulation_steps 16 \
                                    --evaluation_strategy epoch \
                                    --saving_strategy epoch \
                                    --scheduler_type constant \
                                    --log_save_dir $log_save_dir \
                                    --ckpt_save_dir $ckpt_save_dir
        done
    done
done