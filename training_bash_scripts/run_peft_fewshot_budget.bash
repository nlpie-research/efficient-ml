# model_name_or_path=(nlpie/bio-mobilebert
#                     nlpie/tiny-biobert
#                     dmis-lab/biobert-v1.1
#                     roberta-base
#                     nlpie/bio-distilbert-uncased
#                     )
model_name_or_path=(roberta-base)
peft_methods=(LORA Full)
task=mimic-mp
max_epochs=10
few_shot_n=(16 32 64 128 256 512 1024 2048 4096)
gpu=3
log_save_dir=/mnt/sdh/effecient_ml/fewshot_budget/logs
ckpt_save_dir=/mnt/sdh/effecient_ml/fewshot_budget/ckpts
for few_shot in "${few_shot_n[@]}"
do
    for model in "${model_name_or_path[@]}"
    do
        for peft_method in "${peft_methods[@]}"
        do
            export CUDA_VISIBLE_DEVICES="$gpu"

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
                    --evaluation_strategy epoch \
                    --saving_strategy epoch \
                    --scheduler_type linear \
                    --task "$task" \
                    --peft_method $peft_method \
                    --log_save_dir $log_save_dir \
                    --ckpt_save_dir $ckpt_save_dir \
                    --few_shot_n $few_shot \
                    --lora_rank $lora_rank \
                    --lora_alpha $lora_alpha \
                    --lora_dropout $lora_dropout \
                    --learning_rate $learning_rate
                    
            else
                python peft_trainer.py \
                    --model_name_or_path "$model" \
                    --max_epochs "$max_epochs" \
                    --evaluation_strategy epoch \
                    --saving_strategy epoch \
                    --scheduler_type linear \
                    --task "$task" \
                    --peft_method $peft_method \
                    --log_save_dir $log_save_dir \
                    --ckpt_save_dir $ckpt_save_dir \
                    --few_shot_n $few_shot
            fi
        done
    done
done