# model_name_or_path=(nlpie/bio-mobilebert
#                     nlpie/tiny-biobert
#                     nlpie/distil-biobert
#                     dmis-lab/biobert-v1.1
#                     roberta-base
#                     )
model_name_or_path=(meta-llama/Llama-2-7b-hf)
# model_name_or_path=(nlpie/distil-biobert)
peft_method=LORA
task=mimic-mp
max_epochs=10000
time_budget=(43200) # 12 hours = 43200 seconds 
gpu=0
for tb in "${time_budget[@]}"
do
    for tb in "${time_budget[@]}"
    do
        export CUDA_VISIBLE_DEVICES="$gpu"
        
        if [ "$model" == "meta-llama/Llama-2-7b-hf" ]; then
            
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

        else
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
                --evaluation_strategy "steps" \
                --eval_every_steps 200 \
                --task "$task" \
                --peft_method $peft_method \
                --log_save_dir $log_save_dir \
                --ckpt_save_dir $ckpt_save_dir \
                --time_budget $tb

            # python peft_trainer.py \
            #     --model_name_or_path "$model" \
            #     --max_epochs "$max_epochs" \
            #     --evaluation_strategy "steps" \
            #     --eval_every_steps 200 \
            #     --task "$task" \
            #     --peft_method $peft_method \
            #     --log_save_dir $log_save_dir \
            #     --ckpt_save_dir $ckpt_save_dir \
            #     --time_budget $tb \
            #     --lora_rank $lora_rank \
            #     --lora_alpha $lora_alpha \
            #     --lora_dropout $lora_dropout \
            #     --learning_rate $learning_rate
        fi  
        
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
        done
    done
done
# if [ "$model" == "meta-llama/Llama-2-7b-hf" ]; then
            
#     python peft_trainer.py \
#         --model_name_or_path "$model" \
#         --max_epochs "$max_epochs" \
#         --scheduler_type constant \
#         --evaluation_strategy steps \
#         --eval_every_steps 200 \
#         --task "$task" \
#         --peft_method "$peft_method" \
#         --log_save_dir $log_save_dir \
#         --ckpt_save_dir $ckpt_save_dir \
#         --train_batch_size 2 \
#         --eval_batch_size 2 \
#         --learning_rate 0.00005 \
#         --lora_rank 16 \
#         --lora_alpha 32 \
#         --time_budget $tb \
#         --gradient_accumulation_steps 8