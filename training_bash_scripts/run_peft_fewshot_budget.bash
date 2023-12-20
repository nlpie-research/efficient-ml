model_name_or_path=(nlpie/bio-mobilebert
                    nlpie/tiny-biobert
                    nlpie/distil-biobert
                    dmis-lab/biobert-v1.1
                    )
peft_method=Full
task=mimic-mp
max_epochs=10
few_shot_n=(16 32 64 128 256 512 1024)
gpu=0
log_save_dir=/mnt/sdd/efficient_ml_data/saved_models/peft/fewshot_budget/logs
ckpt_save_dir=/mnt/sdd/efficient_ml_data/saved_models/peft/fewshot_budget/ckpts
for few_shot in "${few_shot_n[@]}"
do
    for model in "${model_name_or_path[@]}"
    do
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
        
        export CUDA_VISIBLE_DEVICES="$gpu"
        # python peft_trainer.py \
        #     --model_name_or_path "$model" \
        #     --max_epochs "$max_epochs" \
        #     --evaluation_strategy "epoch" \
        #     --eval_every_steps 200 \
        #     --task "$task" \
        #     --peft_method $peft_method \
        #     --log_save_dir $log_save_dir \
        #     --ckpt_save_dir $ckpt_save_dir \
        #     --lora_rank $lora_rank \
        #     --lora_alpha $lora_alpha \
        #     --lora_dropout $lora_dropout \
        #     --learning_rate $learning_rate \
        #     --few_shot_n $few_shot \
        #     --saving_strategy epoch

        python peft_trainer.py \
            --model_name_or_path "$model" \
            --max_epochs "$max_epochs" \
            --evaluation_strategy "epoch" \
            --eval_every_steps 200 \
            --task "$task" \
            --peft_method $peft_method \
            --log_save_dir $log_save_dir \
            --ckpt_save_dir $ckpt_save_dir \
            --few_shot_n $few_shot \
            --saving_strategy epoch

    done
done