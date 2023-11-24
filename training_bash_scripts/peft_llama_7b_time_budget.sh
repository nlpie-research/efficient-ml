#!/usr/bin/env bash
# decapoda-research/llama-7b-hf ybelkada/falcon-7b-sharded-bf16 tiiuae/falcon-7b-instruct
model_name_or_path=(meta-llama/Llama-2-7b-hf                                                               
                     )
task=mimic-mp
peft_methods=(LORA)
max_epochs=5
gpu=3
time_budget=6000
log_save_dir=/mnt/sdd/efficient_ml_data/saved_models/peft/time_budget_"$time_budget"s/logs
ckpt_save_dir=/mnt/sdd/efficient_ml_data/saved_models/peft/time_budget_"$time_budget"s/ckpts
for model in "${model_name_or_path[@]}"
    do
    for peft_method in "${peft_methods[@]}"
        do
        CUDA_VISIBLE_DEVICES="$gpu" python peft_trainer.py --model_name_or_path "$model" \
                                --max_epochs "$max_epochs" \
                                --training_data_dir "$data_dir" \
                                --eval_data_dir "$eval_data_dir" \
                                --task "$task" \
                                --peft_method "$peft_method" \
                                --train_batch_size 4 \
                                --eval_batch_size 4 \
                                --eight_bit_training \
                                --evaluation_strategy "steps" \
                                --eval_every_steps 200 \
                                --log_save_dir $log_save_dir \
                                --ckpt_save_dir $ckpt_save_dir \
                                --learning_rate 0.00005 \
                                --lora_rank 16 \
                                --lora_alpha 32 \
                                --time_budget $time_budget \
    done
done