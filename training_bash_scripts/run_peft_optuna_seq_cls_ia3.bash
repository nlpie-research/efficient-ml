# model_name_or_path=(nlpie/bio-mobilebert
#                     nlpie/tiny-biobert
#                     roberta-base
#                     nlpie/distil-biobert
#                     dmis-lab/biobert-v1.1)
model=$1
gpu=$2
# lora_ranks=($3)
peft_method="IA3" 
task="mimic-mp"
max_epochs=5
log_save_dir="/mnt/sdd/efficient_ml_data/saved_models/peft/Optuna/logs"
ckpt_save_dir="/mnt/sdd/efficient_ml_data/saved_models/peft/Optuna/ckpts"
export CUDA_VISIBLE_DEVICES="$gpu"
python peft_trainer.py \
--model_name_or_path "$model" \
--max_epochs "$max_epochs" \
--task "$task" \
--peft_method "$peft_method" \
--log_save_dir $log_save_dir \
--ckpt_save_dir $ckpt_save_dir \
--evaluation_strategy steps \
--eval_every_steps 200 \
--optuna

