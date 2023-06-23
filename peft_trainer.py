import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3" #6,7import pandas as pd 2080s = 0,3,5,6,8 Nvidia-smi ids: 0, 3, 5, 6, 8 Actual id: 5,6,7,8,9 

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
)


import evaluate
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed, Trainer, TrainingArguments
from tqdm import tqdm
from loguru import logger as loguru_logger
import numpy as np
import argparse
import yaml
import json
from datetime import datetime
from scipy.special import softmax


'''
Script to train a prefix-tuning model on a given dataset. 

Example usage:
# custom models:
 -> /mnt/sdc/niallt/saved_models/declutr/mimic/few_epoch/mimic-roberta-base/2_anch_2_pos_min_1024/transformer_format/
 -> /mnt/sdc/niallt/saved_models/language_modelling/mimic/roberta-base-mimic-wecho/sampled_250000/08-03-2023--13-06/checkpoint-84000/
 -> /mnt/sdc/niallt/saved_models/language_modelling/mimic/mimic-roberta-base/sampled_250000/22-12-2022--12-45/checkpoint-100000

python peft_trainer.py --task icd9-triage-no-category-in-text --training_size fewshot --few_shot_n 64 --model_name_or_path /mnt/sdc/niallt/saved_models/declutr/mimic/few_epoch/mimic-roberta-base/2_anch_2_pos_min_1024/transformer_format/

'''

# setup main function
def main():
    parser = argparse.ArgumentParser()
    

    # Required parameters
    parser.add_argument("--training_data_dir",
                        default = "/mnt/sdc/niallt/mimic_iii/processed/HADM_ID_split/icd9-triage/no_category_in_text/",# triage = /mnt/sdc/niallt/mimic_iii/processed/HADM_ID_split/icd9-triage/
                        type=str,
                        help = "The data path containing the dataset to use")
    parser.add_argument("--eval_data_dir",
                        default = "/mnt/sdc/niallt/mimic_iii/processed/HADM_ID_split/icd9-triage/no_category_in_text/",# triage = /mnt/sdc/niallt/mimic_iii/processed/HADM_ID_split/icd9-triage/
                        type=str,
                        help = "The data path containing the dataset to use")
    parser.add_argument("--cache_dir",
                        default = "/mnt/sdc/niallt/.cache/",
                        type=str,
                        help = "The data path to the directory containing the notes and referral data files")

    parser.add_argument("--training_file",
                        default = "train.csv",
                        type=str,
                        help = "The data path to the directory containing the notes and referral data files")
    parser.add_argument("--validation_file",
                        default = "valid.csv",
                        type=str,
                        help = "The default name of the training file")
    parser.add_argument("--test_file",
                        default = "test.csv",
                        type=str,
                        help = "The default name of the test file")

    parser.add_argument("--pretrained_models_dir",
                        default="",
                        type=str,
                        help="The data path to the directory containing local pretrained models from huggingface")


    parser.add_argument("--text_col",
                        default = "text",
                        type=str,
                        help = "col name for the column containing the text")

    parser.add_argument("--log_save_dir",
                        default = "/mnt/sdc/niallt/saved_models/peft_training/logs/",
                        type=str,
                        help = "The data path to save tb log files to"
                        )
    parser.add_argument("--ckpt_save_dir",
                    default = "/mnt/sdc/niallt/saved_models/peft_training/ckpts/",
                    type=str,
                    help = "The data path to save trained ckpts to"
                    )

    parser.add_argument("--max_length",
                        default = 480,
                        type=int,
                        help = "Max tokens to be used in modelling"
                        )
    parser.add_argument("--max_steps",
                        default = 100000,
                        type=int,
                        help = "The max number of training steps before the trainer will terminate")
    parser.add_argument("--warmup_steps",
                        default = 100,
                        type=int,
                        help = "The max number of training steps before the trainer will terminate")
    parser.add_argument("--eval_accumulation_steps",
                        default = 50,
                        type=int,
                        help = """ Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU. If left unset,
                                the whole predictions are accumulated on GPU/TPU before being moved to the CPU""")
    parser.add_argument("--eval_every_steps",
                        default = 100,
                        type=int,
                        help = "How many steps of training before an evaluation is run on the validation set")
    parser.add_argument("--save_every_steps",
                        default = 100,
                        type=int,
                        help = "How many steps of training before an evaluation is run on the validation set")
    parser.add_argument("--log_every_steps",
                        default = 10,
                        type=int,
                        help = "How often are we logging?")
    parser.add_argument("--train_batch_size",
                        default = 32,
                        type=int,
                        help = "the size of training batches")
    parser.add_argument("--eval_batch_size",
                        default = 32,
                        type=int,
                        help = "the size of evaluation batches")
    parser.add_argument("--max_epochs",
                        default = 1,
                        type=int,
                        help = "the maximum number of epochs to train for")
    parser.add_argument("--accumulate_grad_batches",
                        default = 1,
                        type=int,
                        help = "number of batches to accumlate before optimization step"
                        )

    parser.add_argument("--gpu_idx", 
                        type=int,
                        default=6,
                        help="Which gpu device to use e.g. 0 for cuda:0, or for more gpus use comma separated e.g. 0,1,2")

    parser.add_argument(
            "--model_name_or_path",
            default= "/mnt/sdc/niallt/saved_models/language_modelling/mimic/mimic-roberta-base/sampled_250000/22-12-2022--12-45/checkpoint-100000/",# 'allenai/biomed_roberta_base',#'simonlevine/biomed_roberta_base-4096-speedfix', # 'bert-base-uncased', #emilyalsentzer/Bio_ClinicalBERT'
            type=str,
            help="Encoder model to be used.",
        )    

    parser.add_argument(
        "--encoder_learning_rate",
        default=1e-05,
        type=float,
        help="Encoder specific learning rate.",
    )
    parser.add_argument(
        "--classifier_learning_rate",
        default=1e-05,
        type=float,
        help="Classification head learning rate.",
    )

    parser.add_argument(
        "--dropout",
        default=0.1,
        type=float,
        help="Dropout value for classifier head.",
    )

    parser.add_argument(
        "--task",
        default="mimic-note-category", # icd9_triage
        type=str,
        help="name of dataset",
    )
    parser.add_argument(
        "--task_type",
        default="SEQ_CLS", # SEQ-CLS
        type=str,
        help="name of dataset",
    )
    parser.add_argument(
        "--evaluation_strategy",
        default="epoch", # steps or epoch
        type=str,
        help="Whether to log every n steps or per epoch",
    )
    parser.add_argument(
        "--saving_strategy",
        default="steps", # steps or epoch or no
        type=str,
        help="Whether to save checkpoints and if so how often",
    )      
    parser.add_argument(
        "--model_type",
        default="mean_embedder", 
        choices = ["automodelforsequence","mean_embedder"],
        type=str,
        help="This will alter the architecture and forward pass used by transformer sequence classifier. Autosequence will use default class from Transformers library, custom will use our own with adjustments to forward pass",
    )

    parser.add_argument(
        "--label_col",
        default="label", # label column of dataframes provided - should be label if using the dataprocessors from utils
        type=str,
        help="string value of column name with the int class labels",
    )

    parser.add_argument(
        "--loader_workers",
        default=24,
        type=int,
        help="How many subprocesses to use for data loading. 0 means that \
            the data will be loaded in the main process.",
    )

    # Early Stopping
    parser.add_argument(
        "--monitor", default="monitor_balanced_accuracy", type=str, help="Quantity to monitor."
    )

    parser.add_argument(
        "--metric_mode",
        default="max",
        type=str,
        help="If we want to min/max the monitored quantity.",
        choices=["auto", "min", "max"],
    )
    parser.add_argument(
        "--patience",
        default=4,
        type=int,
        help=(
            "Number of epochs with no improvement "
            "after which training will be stopped."
        ),
    )
    parser.add_argument(
        '--fast_dev_run',
        default=False,
        type=bool,
        help='Run for a trivial single batch and single epoch.'
    )

    parser.add_argument(
        "--optimizer",
        default="adamw",
        type=str,
        help="Optimization algorithm to use e.g. adamw, adafactor"
    )

    parser.add_argument(
        "--training_size",
        default="full",
        type=str,
        help="full training used, fewshot, or zero"
    )   
    parser.add_argument(
        "--peft_method",
        default="LORA", # LORA, PREFIX_TUNING, PROMPT_TUNING, P_TUNING
        type=str,
        help="Which peft method to use"
    )   

    parser.add_argument(
        "--few_shot_n",
        type=int,
        default = None
    )
    parser.add_argument(
        "--eval_few_shot_n",
        type=int,
        default = 128
    )
    
    parser.add_argument(
        '--combined_val_test_sets',
        default=False,
        type=bool,
        help='Whether or not to combine the validation and test datasets'
    )
    parser.add_argument(
        '--sensitivity',
        default=False,
        type=bool,
        help='Run sensitivity trials - investigating the influence of number of transformer layers.'
    )


    parser.add_argument(
        '--no_cuda',
        action = "store_true",        
        help='Whether to use cuda/gpu or just use CPU '
    )

    parser.add_argument('--task_to_keys',
        default = {
                "cola": ("sentence", None),
                "mnli": ("premise", "hypothesis"),
                "mnli-mm": ("premise", "hypothesis"),
                "mrpc": ("sentence1", "sentence2"),
                "qnli": ("question", "sentence"),
                "qqp": ("question1", "question2"),
                "rte": ("sentence1", "sentence2"),
                "sst2": ("sentence", None),
                "stsb": ("sentence1", "sentence2"),
                "wnli": ("sentence1", "sentence2"),
                "mimic-note-category": ("TEXT", None),
                "icd9-triage":("text", None),
                "icd9-triage-no-category-in-text":("text", None),
                },
        type = dict,
        help = "mapping of task name to tuple of the note formats"
    )

    # TODO - add an argument to specify whether using balanced data then update directories based on that
    args = parser.parse_args()
    
    # setup params
    training_data_dir = args.training_data_dir
    eval_data_dir = args.eval_data_dir
    cache_dir = args.cache_dir
    training_file = args.training_file
    validation_file = args.validation_file
    test_file = args.test_file
    text_col = args.text_col
    log_save_dir = args.log_save_dir
    ckpt_save_dir = args.ckpt_save_dir
    peft_method = args.peft_method
    task = args.task
    task_type = args.task_type
    task_to_keys = args.task_to_keys
    pretrained_models_dir = args.pretrained_models_dir
    model_name_or_path = args.model_name_or_path
    max_length = args.max_length
    few_shot_n = args.few_shot_n
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    num_epochs = args.max_epochs
    
    # set up some variables to add to checkpoint and logs filenames
    time_now = str(datetime.now().strftime("%d-%m-%Y--%H-%M"))
    
    
    
    # define some model specific params for logs etc - this is mainly for custom local models
    # TODO clean this up/improve
    # THIS IS ALL VERY CRUDE AND DEPENDENT ON HAVING TRAINED USING THE SCRIPTS INSIDE THIS REPO - forward slashes really matter for the naming convention make sure to append the path with a forward slash
    if "saved_models" in model_name_or_path:
        if "declutr" in model_name_or_path:
            if "few_epoch" in model_name_or_path:
                if "span_comparison" in model_name_or_path:
                    model_name = model_name_or_path.split("/")[9] + "/declutr/" + model_name_or_path.split("/")[-3]
                else:
                    model_name = model_name_or_path.split("/")[8] + "/declutr/" + model_name_or_path.split("/")[-3]

            else:
                model_name = model_name_or_path.split("/")[7] + "/declutr/" + model_name_or_path.split("/")[-3]
        elif "contrastive" in model_name_or_path or "custom_pretraining" in model_name_or_path:
            model_name = model_name_or_path.split("/")[7]
        elif "simcse" in model_name_or_path:# change this to be dynamic
            model_name = "simcse-mimic"
        else:
            model_name = model_name_or_path.split("/")[7]
    else:    
        model_name = model_name_or_path.split("/")[-1]
        
    
     
    # set up logging and ckpt dirs
    logging_dir = f"{log_save_dir}/{task}/fewshot_{few_shot_n}/{model_name}/{peft_method}/{time_now}/"
    ckpt_dir = f"{ckpt_save_dir}/{task}/fewshot_{few_shot_n}/{model_name}/{peft_method}/{time_now}/"
    
    loguru_logger.info(f"Logging to: {logging_dir}")
    loguru_logger.info(f"Saving ckpts to: {ckpt_dir}")  
    
    # update training data dir based on few shot
    if args.training_size == "fewshot":
        training_data_dir = f"{training_data_dir}/fewshot_{few_shot_n}/"
        loguru_logger.info(f"Training data dir updated to: {training_data_dir}") 
    
        
    if args.saving_strategy != "no":
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        
    
    
    # set up device based on whether cuda is available
    if args.no_cuda:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"    
    
    
    ######################### Data setup #########################
    loguru_logger.info("Loading data")
    # setup datasets and metric
    datasets = load_dataset("csv", 
                        data_files = {"train":f"{training_data_dir}/{training_file}",
                                        "validation":f"{eval_data_dir}/{validation_file}",
                                        "test":f"{eval_data_dir}/{test_file}"},
                        cache_dir = cache_dir)
    
    loguru_logger.info(f"Data loaded. Dataset info: {datasets}")
    
    # get number of labels
    num_labels = len(np.unique(datasets["train"]["label"]))
    loguru_logger.info(f"Number of labels is: {num_labels}")
    
    if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
        padding_side = "left"
    else:
        padding_side = "right"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    print(f"tokenizer is: {tokenizer}")

    # set up data keys
    sentence1_key, sentence2_key = task_to_keys[task]
    # tokenize function
    def tokenize_function(examples):
        # max_length is important when using prompt tuning  or prefix tuning or p tuning as virtual tokens are added - which can overshoot the max length in pefts current form
        # for now set to 480 and see how it goes
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], truncation=True, max_length = args.max_length)
        return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True, max_length=args.max_length)
    
    
    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], truncation=True)
        return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

    # create encoded dataset
    # encoded_dataset = datasets.map(preprocess_function, batched=True)
    # encoded_dataset = encoded_dataset.rename_column("label", "labels") # this seems needed for the peft models...

    # print(encoded_dataset)
    


    # # apply to dataset
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=['text', 'triage-category'],
    )

    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    # loguru_logger.info(f"Tokenized datasets: {tokenized_datasets}")

    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")


    # # Instantiate dataloaders.
    # train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=train_batch_size)
    # eval_dataloader = DataLoader(
    #     tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=train_batch_size
    # )
    
    # metrics function for trainer
    def compute_metrics(eval_pred):
        
        precision_score = evaluate.load("precision")
        recall_score = evaluate.load("recall")
        accuracy_score = evaluate.load("accuracy")
        f1_score = evaluate.load("f1")        
        roc_auc_score = evaluate.load("roc_auc", "multiclass")        

        logits, labels = eval_pred
        
        # print(f"logits are: {logits} of shape: {logits.shape}")
        #TODO add softmax to convert logits to probs
        # print(f"logits shape is: {logits.shape}")
        pred_scores = softmax(logits, axis = -1)        
        predictions = np.argmax(logits, axis = -1)
        
        # print(f"Labels are: {labels}\n")
        # print(f"Preds are: {predictions}")
        precision = precision_score.compute(predictions=predictions, references=labels, average = "macro")["precision"]
        recall = recall_score.compute(predictions=predictions, references=labels, average = "macro")["recall"]
        accuracy = accuracy_score.compute(predictions=predictions, references=labels)["accuracy"]
        f1_macro = f1_score.compute(predictions=predictions, references=labels, average = "macro")["f1"]
        f1_weighted = f1_score.compute(predictions=predictions, references=labels, average = "weighted")["f1"]
        # roc_auc has slightly different format - needs the probs/scores rather than predicted labels
        roc_auc = roc_auc_score.compute(references=labels,
                                        prediction_scores = pred_scores,
                                        multi_class = 'ovr', 
                                        average = "macro")['roc_auc']
        
        return {"precision": precision, 
                "recall": recall,
                "accuracy": accuracy,
                "f1_macro":f1_macro,
                "f1_weighted":f1_weighted,
                "roc_auc_macro":roc_auc}
    
    
    ######################### Model setup #########################
    loguru_logger.info("Setting up model")
    # set up some PEFT params
    
    if peft_method == "LORA":
        loguru_logger.info("Using LORA")
        peft_type = PeftType.LORA
        lr = 3e-4
        peft_config = LoraConfig(task_type=task_type, inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
    elif peft_method == "PREFIX_TUNING":
        loguru_logger.info("Using PREFIX_TUNING")
        peft_type = PeftType.PREFIX_TUNING
        peft_config = PrefixTuningConfig(task_type=task_type, num_virtual_tokens=20)
        lr = 1e-2
    elif peft_method == "PROMPT_TUNING":
        loguru_logger.info("Using PROMPT_TUNING")
        peft_type = PeftType.PROMPT_TUNING
        peft_config = PromptTuningConfig(task_type=task_type, num_virtual_tokens=10)
        lr = 1e-3
    elif peft_method == "P_TUNING":
        loguru_logger.info("Using P_TUNING")
        peft_type = PeftType.P_TUNING
        peft_config = PromptEncoderConfig(task_type=task_type, num_virtual_tokens=20, encoder_hidden_size=128)
        lr = 1e-3
        

    # load peft model    
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path,
                                                                num_labels = num_labels,
                                                                output_hidden_states=False, # if true will lead to GPU OOM error
                                                                return_dict=True)
    
    #FIXME - at moment when using custom roberta it ledas to GPU OOM error during evaluation loop
    #########uncomment below for PEFT models#########
    model = get_peft_model(model, peft_config)
    print(f"peft config is: {peft_config}")
    print(model)
    model.print_trainable_parameters()
    
       
    model.to(device)
    
    # setup optimizer and lr_scheduler
    optimizer = AdamW(params=model.parameters(), lr=lr)    

    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0.06 * (len(datasets['train'])/train_batch_size * num_epochs),
        num_training_steps=(len(datasets['train'])/train_batch_size * num_epochs),
    )

    
    
    ######################### Trainer setup #########################
    # set up trainer arguments

    monitor_metric_name = "f1_macro"


    train_args = TrainingArguments(
        output_dir = f"{ckpt_dir}/",
        evaluation_strategy = args.evaluation_strategy,
        eval_steps = args.eval_every_steps,
        logging_steps = args.log_every_steps,
        logging_first_step = True,    
        save_strategy = args.saving_strategy,
        save_steps = args.save_every_steps,        
        per_device_train_batch_size = train_batch_size,
        per_device_eval_batch_size = eval_batch_size,
        num_train_epochs=args.max_epochs,
        weight_decay=0.01,
        load_best_model_at_end=False,
        metric_for_best_model=monitor_metric_name,
        push_to_hub=False,
        logging_dir = f"{logging_dir}/",
        save_total_limit=2,
        report_to = 'tensorboard',        
        overwrite_output_dir=True,
        # eval_accumulation_steps=32,         
        # will avoid building up lots of files
        no_cuda = args.no_cuda, # for cpu only
        # use_ipex = args.use_ipex # for cpu only
        # remove_unused_columns=False, # at moment the peft model changes the output format and this causes issues with the trainer
        # label_names = ["labels"],#FIXME - this is a hack to get around the fact that the peft model changes the output format and this causes issues with the trainer
    )
    
    # setup normal trainer
    trainer = Trainer(
        model,
        train_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,         
        data_collator= collate_fn,
        optimizers = (optimizer, lr_scheduler),
        )
    # run training
    trainer.train()
    
    # run evaluation on test set
    trainer.evaluate()
    # save the args/params to a text/yaml file
    with open(f'{logging_dir}/config.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        
    with open(f'{logging_dir}/config.yaml', 'w') as f:
        yaml.dump(args.__dict__, f) 
    # also save trainer args
    with open(f'{logging_dir}/all_trainer_args.yaml', 'w') as f:
        yaml.dump(trainer.args.__dict__, f)   
        
    # save the peft weights to a file
    model.save_pretrained(f"{ckpt_dir}")

# run script
if __name__ == "__main__":
    main()