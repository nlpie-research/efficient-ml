# model_name_or_path=(nlpie/bio-mobilebert
#                     nlpie/tiny-biobert
#                     nlpie/distil-biobert
#                     dmis-lab/biobert-v1.1
#                     )
# model_name_or_path=(meta-llama/Llama-2-7b-hf)
model_name_or_path=(nlpie/distil-biobert)
peft_method=Full
task=mimic-mp
max_epochs=10000
time_budget=(2000)
gpu=1
for tb in "${time_budget[@]}"
do
    log_save_dir=/mnt/sdd/efficient_ml_data/saved_models/peft/tb_${tb}/logs
    ckpt_save_dir=/mnt/sdd/efficient_ml_data/saved_models/peft/tb_${tb}/ckpts
    for model in "${model_name_or_path[@]}"
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
                --train_batch_size 2 \
                --eval_batch_size 2 \
                --learning_rate 0.00005 \
                --lora_rank 16 \
                --lora_alpha 32 \
                --time_budget $tb \
                --gradient_accumulation_steps 8

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
        
    done
done