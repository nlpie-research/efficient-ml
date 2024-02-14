# model_name_or_path=(nlpie/bio-mobilebert
#                     nlpie/tiny-biobert
#                     nlpie/distil-biobert
#                     dmis-lab/biobert-v1.1
#                     roberta-base
#                     )
model_name_or_path=(roberta-base)
peft_methods=(LORA Full)
task=mimic-mp
max_epochs=6
time_budget=(6000) # 12 hours = 43200 seconds 
gpu=0
log_save_dir=/mnt/sdh/effecient_ml/tb_2000/logs
ckpt_save_dir=/mnt/sdh/effecient_ml/tb_2000/ckpts
for model in "${model_name_or_path[@]}"
do
    for peft_method in "${peft_methods[@]}"
    do
        for tb in "${time_budget[@]}"
        do
            export CUDA_VISIBLE_DEVICES="$gpu"
            
            if [ "$model" == "meta-llama/Llama-2-7b-hf" ]; then
                if [ "$peft_method" == "LORA" ]; then
                    python peft_trainer.py \
                        --model_name_or_path "$model" \
                        --max_epochs "$max_epochs" \
                        --evaluation_strategy "steps" \
                        --eval_every_steps 200 \
                        --task "$task" \
                        --peft_method "$peft_method" \
                        --log_save_dir $log_save_dir \
                        --ckpt_save_dir $ckpt_save_dir \
                        --train_batch_size 1 \
                        --eval_batch_size 1 \
                        --learning_rate 0.00005 \
                        --lora_rank 16 \
                        --lora_alpha 32 \
                        --time_budget $tb \
                        --scheduler_type constant \
                        --gradient_accumulation_steps 16
                        # --eight_bit_training
                        # --few_shot_n 10 \
                        # --eval_few_shot_n 10
                        # --eight_bit_training
                fi
            else
                if [ "$peft_method" == "LORA" ]; then
                    # Load best LORA params for model type
                    model_params=$(grep $model /mnt/sdd/efficient_ml_data/optuna_dbs/Runs/best_params_LORA.csv)
                    IFS=',' read -r -a array <<< "$model_params"
                    lora_rank=${array[1]}
                    lora_alpha=${array[2]}
                    lora_dropout=${array[3]}
                    learning_rate=${array[4]}
                    echo Running ${model} with 
                    echo -e "\t"rank=${lora_rank}
                    echo -e "\t"alpha=${lora_alpha}
                    echo -e "\t"dropout=${lora_dropout}
                    echo -e "\t"lr=${learning_rate}
                    echo
                    
                    python peft_trainer.py \
                        --model_name_or_path "$model" \
                        --max_epochs "$max_epochs" \
                        --scheduler_type linear \
                        --evaluation_strategy steps \
                        --eval_every_steps 200 \
                        --task "$task" \
                        --peft_method $peft_method \
                        --log_save_dir $log_save_dir \
                        --ckpt_save_dir $ckpt_save_dir \
                        --time_budget $tb \
                        --lora_rank $lora_rank \
                        --lora_alpha $lora_alpha \
                        --lora_dropout $lora_dropout \
                        --learning_rate $learning_rate
                    
                else
                    python peft_trainer.py \
                        --model_name_or_path "$model" \
                        --max_epochs "$max_epochs" \
                        --scheduler_type linear \
                        --evaluation_strategy steps \
                        --eval_every_steps 200 \
                        --task "$task" \
                        --peft_method $peft_method \
                        --log_save_dir $log_save_dir \
                        --ckpt_save_dir $ckpt_save_dir \
                        --time_budget $tb
                fi  
            fi
        done
    done
done