import os
from functools import partial

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
# 6,7
# 2080s = 0,3,5,6,8 
# When specifiying from the command line, Nvidia-smi ids: 0, 3, 5, 6, 8 Actual id: 5,6,7,8,9 

import argparse
import json
from datetime import datetime

import evaluate
import numpy as np
import torch
import yaml
from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict
from loguru import logger as loguru_logger
from peft import (LoraConfig, PeftType, PrefixTuningConfig,
                  PromptEncoderConfig, PromptTuningConfig, TaskType,
                  get_peft_config, get_peft_model, get_peft_model_state_dict,
                  prepare_model_for_int8_training,
                  prepare_model_for_kbit_training, set_peft_model_state_dict)
from scipy.special import softmax
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification,
                          AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification,
                          LlamaForSequenceClassification, LlamaTokenizer,
                          Trainer, TrainingArguments,
                          get_linear_schedule_with_warmup, set_seed)

'''
Script to train a prefix-tuning model on a given dataset. 

Example usage:
# custom models:
 -> /mnt/sdc/niallt/saved_models/declutr/mimic/few_epoch/mimic-roberta-base/2_anch_2_pos_min_1024/transformer_format/
 -> /mnt/sdc/niallt/saved_models/language_modelling/mimic/roberta-base-mimic-wecho/sampled_250000/08-03-2023--13-06/checkpoint-84000/
 -> /mnt/sdc/niallt/saved_models/language_modelling/mimic/mimic-roberta-base/sampled_250000/22-12-2022--12-45/checkpoint-100000

python peft_trainer.py 
    --task icd9-triage-no-category-in-text 
    --training_size fewshot 
    --few_shot_n 64 
    --model_name_or_path /mnt/sdc/niallt/saved_models/declutr/mimic/few_epoch/mimic-roberta-base/2_anch_2_pos_min_1024/transformer_format/

'''

class DatasetInfo:
  def __init__(self, name,
               ds_type="ner", 
               metric=None, 
               load_from_disk=True,
               isMultiSentence=False, 
               validationSubsets=["test"],
               lr=[5e-5, 2e-5, 1e-5], 
               batch_size=[32], 
               epochs=3, 
               runs=1,
               num_labels=None):

    self.name = name
    self.isMultiSentence = isMultiSentence
    self.validationSubsets = validationSubsets
    self.lr = lr
    self.batch_size = batch_size
    self.epochs = epochs
    self.runs = runs
    self.load_from_disk = load_from_disk
    self.type = ds_type
    self.num_labels = num_labels

    if metric == None:
      self.metric = "accuracy"
    else:
      self.metric = metric

    self.fullName = name + "-" + self.metric

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--data_dir",
                        default = "",
                        type=str,
                        help = "The data path containing the dataset to use. These datasets use the load_datasets and DatasetInfo method to load the data")
    parser.add_argument("--training_data_dir",
                        default = "/mnt/sdc/niallt/mimic_iii/processed/HADM_ID_split/icd9-triage/no_category_in_text",# triage = /mnt/sdc/niallt/mimic_iii/processed/HADM_ID_split/icd9-triage/
                        type=str,
                        help = "The data path containing the dataset to use")
    parser.add_argument("--eval_data_dir",
                        default = "/mnt/sdc/niallt/mimic_iii/processed/HADM_ID_split/icd9-triage/no_category_in_text",# triage = /mnt/sdc/niallt/mimic_iii/processed/HADM_ID_split/icd9-triage/
                        type=str,
                        help = "The data path containing the dataset to use")
    parser.add_argument("--cache_dir",
                        default = None,
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
                        default = "/mnt/sdc/niallt/saved_models/peft_training/logs",
                        type=str,
                        help = "The data path to save tb log files to")
    parser.add_argument("--ckpt_save_dir",
                        default = "/mnt/sdc/niallt/saved_models/peft_training/ckpts",
                        type=str,
                        help = "The data path to save trained ckpts to")
    parser.add_argument("--max_length",
                        default = 480,
                        type=int,
                        help = "Max tokens to be used in modelling")
    parser.add_argument("--warmup_steps",
                        default = 100,
                        type=int,
                        help = "The max number of training steps before the trainer will terminate")
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
    parser.add_argument("--model_name_or_path",
                        default= "/mnt/sdc/niallt/saved_models/language_modelling/mimic/mimic-roberta-base/sampled_250000/22-12-2022--12-45/checkpoint-100000",
                        # 'allenai/biomed_roberta_base',#'simonlevine/biomed_roberta_base-4096-speedfix', # 'bert-base-uncased', #emilyalsentzer/Bio_ClinicalBERT'
                        type=str,
                        help="Encoder model to be used.")
    parser.add_argument("--dropout",
                        default=0.1,
                        type=float,
                        help="Dropout value for classifier head.")
    parser.add_argument("--task",
                        default="mimic-note-category", # icd9_triage
                        type=str,
                        help="name of dataset")
    parser.add_argument("--task_type",
                        default="SEQ_CLS", # SEQ-CLS
                        choices=["SEQ_CLS", "TOKEN_CLS"],
                        type=str,
                        help="name of dataset")
    parser.add_argument("--evaluation_strategy",
                        default="epoch", # steps or epoch
                        type=str,
                        help="Whether to log every n steps or per epoch")
    parser.add_argument("--saving_strategy",
                        default="no", # steps or epoch or no
                        type=str,
                        help="Whether to save checkpoints and if so how often")      
    parser.add_argument("--label_col",
                        default="label", # label column of dataframes provided - should be label if using the dataprocessors from utils
                        type=str,
                        help="string value of column name with the int class labels")
    parser.add_argument("--loader_workers",
                        default=24,
                        type=int,
                        help="How many subprocesses to use for data loading. 0 means that \
                            the data will be loaded in the main process.")
    parser.add_argument("--monitor", 
                        default="monitor_balanced_accuracy", 
                        type=str, 
                        help="Quantity to monitor.")

    parser.add_argument("--metric_mode",
                        default="max",
                        type=str,
                        help="If we want to min/max the monitored quantity.",
                        choices=["auto", "min", "max"])
    parser.add_argument("--patience",
                        default=4,
                        type=int,
                        help=(
                            "Number of epochs with no improvement "
                            "after which training will be stopped."
                        ))
    parser.add_argument('--fast_dev_run',
                        default=False,
                        type=bool,
                        help='Run for a trivial single batch and single epoch.')

    parser.add_argument("--optimizer",
                        default="adamw",
                        type=str,
                        help="Optimization algorithm to use e.g. adamw, adafactor")

    parser.add_argument("--peft_method",
                        default="LORA", # LORA, PREFIX_TUNING, PROMPT_TUNING, P_TUNING
                        type=str,
                        help="Which peft method to use")   
    parser.add_argument("--few_shot_n",
                        type=int,
                        default = None)
    parser.add_argument("--eval_few_shot_n",
                        type=int,
                        default = 128)
    
    parser.add_argument('--combined_val_test_sets',
                        default=False,
                        type=bool,
                        help='Whether or not to combine the validation and test datasets')
    parser.add_argument('--sensitivity',
                        default=False,
                        type=bool,
                        help='Run sensitivity trials - investigating the influence of number of transformer layers.')
    parser.add_argument('--no_cuda',
                        action = "store_true",        
                        help='Whether to use cuda/gpu or just use CPU ')

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
                                "mimic-los": ("text", None)
                                },
                        type = dict,
                        help = "mapping of task name to tuple of the note formats")
    parser.add_argument(
        '--eight_bit_training',
        action = "store_true",               
        help='Whether to run in 8bit - very needed for llama 7b etc'
    )

    args = parser.parse_args()
    return args

def get_dataset_columns(key:str, task_to_keys:dict) -> tuple[str,str]:
    """
    Get tuple for key. 
    If key is not found, return default tuple.
    """
    
    if key in task_to_keys:
        return task_to_keys[key]
    else:
        return ('text', None)

def get_model_name(model_name_or_path:str) -> str:
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
    
    return model_name

def get_dataset_directory_details(args:argparse.Namespace) -> argparse.Namespace:

    with open('datasets.yaml', 'r') as f:
        datasets = yaml.load(f, yaml.FullLoader)
    
    try:
        dataset_info = datasets[args.task]
        for k, v in dataset_info.items():
            setattr(args, k, v)
    except KeyError:
        print(f"Task name {args.task} not in datasets.yaml. Available tasks are: {list(datasets.keys())}")
        exit(0)

    return args

def load_dataset_from_csv(args:argparse.Namespace, tokenizer:AutoTokenizer) -> tuple:
    eval_data_dir = args.eval_data_dir
    cache_dir = args.cache_dir
    training_file = args.training_file
    validation_file = args.validation_file
    test_file = args.test_file
    task_to_keys = args.task_to_keys
    task = args.task
    training_data_dir = args.training_data_dir
    training_data_dir = args.training_data_dir
    task = args.task
    model_name_or_path = args.model_name_or_path
    few_shot_n = args.few_shot_n
    remove_coumns = args.remove_columns
    

        
    #TODO - add ability to do fewshot sampling for using datasets directly - rather than 
    # loading in separate csv folders
    
    def tokenize_function(examples):
        # max_length is important when using prompt tuning  or prefix tuning 
        # or p tuning as virtual tokens are added - which can overshoot the 
        # max length in pefts current form
        # for now set to 480 and see how it goes
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], 
                             truncation=True, 
                             max_length = args.max_length)
        return tokenizer(examples[sentence1_key], examples[sentence2_key], 
                         truncation=True, max_length=args.max_length)
      
    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], truncation=True)
        return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

    datasets = load_dataset("csv", 
                        data_files = {"train":f"{training_data_dir}/{training_file}",
                                        "validation":f"{eval_data_dir}/{validation_file}",
                                        "test":f"{eval_data_dir}/{test_file}"},
                        cache_dir = cache_dir)
    
    loguru_logger.info(f"Data loaded. Dataset info: {datasets}")
    
    # get number of labels
    num_labels = len(np.unique(datasets["train"][args.label_name]))
    loguru_logger.info(f"Number of labels is: {num_labels}")
    
    print(f"tokenizer is: {tokenizer}")

    # set up data keys
    sentence1_key, sentence2_key = get_dataset_columns(task, task_to_keys)
    # tokenize function
    
    # # apply to dataset
    #FIXME - this is only valid for triage task - need to make more general
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=args.remove_columns)

    return tokenized_datasets, num_labels

def load_datasets(info:DatasetInfo, tokenizer:AutoTokenizer) -> tuple:
    """#Dataset Utilities"""
    
    if not info.load_from_disk:
      dataset = load_dataset(info.name)
    else:
      dataset = load_from_disk(info.name)

    if info.type == "classification":
      num_labels = len(set(dataset["train"]["labels"]))
      def mappingFunction(samples, **kargs):
        if info.isMultiSentence:
          outputs = tokenizer(samples[dataset["train"].column_names[0]],
                              samples[dataset["train"].column_names[1]],
                              max_length=512,
                              truncation=True,
                              padding=kargs["padding"])
        else:
          outputs = tokenizer(samples[dataset["train"].column_names[0]],
                              truncation=True,
                              max_length=512,
                              padding=kargs["padding"])

        outputs["labels"] = samples["labels"]

        return outputs
    elif info.type == "ner":
      # print(dataset)
      num_labels = len(dataset["info"][0]["all_ner_tags"])
      def mappingFunction(all_samples_per_split, **kargs):
        tokenized_samples = tokenizer.batch_encode_plus(all_samples_per_split["tokens"],
                                                        is_split_into_words=True, 
                                                        truncation=True,
                                                        max_length=512,
                                                        padding=kargs["padding"])  
        total_adjusted_labels = []

        for k in range(0, len(tokenized_samples["input_ids"])):
            prev_wid = -1
            word_ids_list = tokenized_samples.word_ids(batch_index=k)
            existing_label_ids = all_samples_per_split["ner_tags"][k]
            i = -1
            adjusted_label_ids = []

            for wid in word_ids_list:
                if(wid is None):
                    adjusted_label_ids.append(-100)
                elif(wid!=prev_wid):
                    i = i + 1
                    adjusted_label_ids.append(existing_label_ids[i])
                    prev_wid = wid
                else:
                    adjusted_label_ids.append(existing_label_ids[i])
                
            total_adjusted_labels.append(adjusted_label_ids)

        tokenized_samples["labels"] = total_adjusted_labels
        
        return tokenized_samples

    tokenizedTrainDataset = dataset["train"].map(mappingFunction,
                                                batched=True,
                                                remove_columns=dataset["train"].column_names,
                                                fn_kwargs={"padding": "do_not_pad"})
      
    tokenizedValDatasets = []

    for name in info.validationSubsets:
      tokenizedValDataset = dataset[name].map(mappingFunction,
                                              batched=True,
                                              remove_columns=dataset[name].column_names,
                                              fn_kwargs={"padding": "do_not_pad"})
      
      tokenizedValDatasets.append(tokenizedValDataset)

    if info.num_labels != None:
      num_labels = info.num_labels

    return tokenizedTrainDataset, tokenizedValDatasets, num_labels, dataset["info"][0]["all_ner_tags"]

def compute_token_cls_metrics(eval_pred, label_list, metric):
    predictions, labels = eval_pred
    # print(f"logits shape: {predictions.shape}")
    # print(f"labels: {labels}")
    predictions = np.argmax(predictions, axis=2)
    # print(f"predictions: {predictions}")
    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    # print(f"True predictions: {true_predictions}")
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    # print(f"True labels: {true_labels}")
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def compute_seq_cls_metrics(eval_pred):
    precision_score = evaluate.load("precision")
    recall_score = evaluate.load("recall")
    accuracy_score = evaluate.load("accuracy")
    f1_score = evaluate.load("f1")        
          

    logits, labels = eval_pred    
    # print(f"logits shape is: {logits.shape}")
    pred_scores = softmax(logits, axis = -1)        
    predictions = np.argmax(logits, axis = -1)
    
    # roc_auc_score needs to handle both binary and multiclass
    # check shape of logits to determine which to use
    if logits.shape[1] == 1 or logits.shape[1] == 2:
        roc_auc_score = evaluate.load("roc_auc", "binary")
        roc_auc = roc_auc_score.compute(references=labels,
                                        # just take the probabilties of the positive class
                                        prediction_scores = pred_scores[:,1]                                         
                                        )['roc_auc']
    else:
        roc_auc_score = evaluate.load("roc_auc", "multiclass")

        roc_auc = roc_auc_score.compute(references=labels,
                                        prediction_scores = pred_scores,
                                        multi_class = 'ovr', 
                                        average = "macro")['roc_auc']  
    # print(f"logits are: {logits} of shape: {logits.shape}")
    
    # print(f"Labels are: {labels}\n")
    # print(f"Preds are: {predictions}")
    precision = precision_score.compute(predictions=predictions, references=labels, average = "macro")["precision"]
    recall = recall_score.compute(predictions=predictions, references=labels, average = "macro")["recall"]
    accuracy = accuracy_score.compute(predictions=predictions, references=labels)["accuracy"]
    f1_macro = f1_score.compute(predictions=predictions, references=labels, average = "macro")["f1"]
    f1_weighted = f1_score.compute(predictions=predictions, references=labels, average = "weighted")["f1"]

    
    return {"precision": precision, 
            "recall": recall,
            "accuracy": accuracy,
            "f1_macro":f1_macro,
            "f1_weighted":f1_weighted,
            "roc_auc_macro":roc_auc}

def create_peft_config(peft_method:str, model_name_or_path:str, task_type:str) -> tuple:
    if peft_method == "LORA":
        loguru_logger.info("Using LORA")
        # according to authors for falcon model adding all linear layers is important     
        if "falcon" in model_name_or_path:
            loguru_logger.info("Using falcon config")
            lora_alpha = 16
            lora_dropout = 0.1
            lora_r = 64

            peft_config = LoraConfig(
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                r=lora_r,
                bias="none",
                inference_mode=False,
                task_type=task_type,
                target_modules=[
                    "query_key_value",
                    "dense",
                    "dense_h_to_4h",
                    "dense_4h_to_h",
                ]
            )
            lr = 3e-4
        else:
            peft_type = PeftType.LORA
            lr = 3e-4
            peft_config = LoraConfig(task_type=task_type, inference_mode=False, 
                                    r=8, lora_alpha=16, lora_dropout=0.1)
    elif peft_method == "PREFIX_TUNING":
        loguru_logger.info("Using PREFIX_TUNING")
        peft_type = PeftType.PREFIX_TUNING
        peft_config = PrefixTuningConfig(task_type=task_type, 
                                         num_virtual_tokens=20)
        lr = 1e-2
    elif peft_method == "PROMPT_TUNING":
        loguru_logger.info("Using PROMPT_TUNING")
        peft_type = PeftType.PROMPT_TUNING
        peft_config = PromptTuningConfig(task_type=task_type, 
                                         num_virtual_tokens=10)
        lr = 1e-3
    elif peft_method == "P_TUNING":
        loguru_logger.info("Using P_TUNING")
        peft_type = PeftType.P_TUNING
        peft_config = PromptEncoderConfig(task_type=task_type, 
                                          num_virtual_tokens=20, 
                                          encoder_hidden_size=128)
        lr = 1e-3

    return peft_config, lr

# setup main function
def main() -> None:
    args = parse_args()    
    args = get_dataset_directory_details(args)

    # setup params
    data_dir = args.data_dir
    log_save_dir = args.log_save_dir
    ckpt_save_dir = args.ckpt_save_dir
    peft_method = args.peft_method
    task = args.task
    task_type = args.task_type
    model_name_or_path = args.model_name_or_path
    few_shot_n = args.few_shot_n
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    num_epochs = args.max_epochs
    
    # set up some variables to add to checkpoint and logs filenames
    time_now = str(datetime.now().strftime("%d-%m-%Y--%H-%M"))
    
    # define some model specific params for logs etc - this is mainly for custom local models
    # TODO clean this up/improve
    # THIS IS ALL VERY CRUDE AND DEPENDENT ON HAVING TRAINED USING THE SCRIPTS INSIDE THIS REPO - 
    # forward slashes really matter for the naming convention make sure to append the path with a forward slash
    model_name = get_model_name(model_name_or_path)
    
    # set up logging and ckpt dirs
    if few_shot_n is not None:
        logging_dir = f"{log_save_dir}/{task}/fewshot_{few_shot_n}/{model_name}/{peft_method}/{time_now}/"
        ckpt_dir = f"{ckpt_save_dir}/{task}/fewshot_{few_shot_n}/{model_name}/{peft_method}/{time_now}/"
    else:
        logging_dir = f"{log_save_dir}/{task}/full/{model_name}/{peft_method}/{time_now}/"
        ckpt_dir = f"{ckpt_save_dir}/{task}/full/{model_name}/{peft_method}/{time_now}/"
    
    loguru_logger.info(f"Logging to: {logging_dir}")
    loguru_logger.info(f"Saving ckpts to: {ckpt_dir}")  
    
    if args.saving_strategy != "no":
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        
    # set up device based on whether cuda is available
    if args.no_cuda:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"    
    
    ######################### Load Tokenizer #########################
    loguru_logger.info("Loading Tokenizer")
    tokenizer_args = {"pretrained_model_name_or_path":model_name_or_path}
    if task_type == "SEQ_CLS":
        if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
            tokenizer_args["padding_side"] = "left"
        else:
            tokenizer_args["padding_side"] = "right"
    else:
        if "roberta" in model_name_or_path:
            tokenizer_args["add_prefix_space"] = True
            
    if "llama" in model_name_or_path:
        loguru_logger.info("Using llama tokenizer")
        tokenizer = LlamaTokenizer.from_pretrained(**tokenizer_args)
    else:
        tokenizer = AutoTokenizer.from_pretrained(**tokenizer_args)   

    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    if task_type == "SEQ_CLS":
        def collate_fn(examples):
            return tokenizer.pad(examples, padding="longest", return_tensors="pt")
    else:    
        collate_fn = DataCollatorForTokenClassification(tokenizer)

    ######################### Data setup #########################
    loguru_logger.info("Loading data")
    if len(data_dir) == 0:
       tokenized_datasets, num_labels = load_dataset_from_csv(args, tokenizer)
    else:
        if task_type == "SEQ_CLS":
            ds_type = "classification"
        else:
            ds_type = "ner"
        ds_info = DatasetInfo(name=data_dir, 
                    metric="f1", load_from_disk=True,
                    ds_type=ds_type, isMultiSentence=False,
                    lr=[5e-5, 2e-5, 1e-5], epochs=3,
                    batch_size=[train_batch_size],
                    runs=1)
        data_tuple = load_datasets(info=ds_info, tokenizer=tokenizer)
        tokenized_datasets = DatasetDict()
        tokenized_datasets["train"] = data_tuple[0]
        tokenized_datasets["validation"] = data_tuple[1][0]
        num_labels = data_tuple[2]
        all_ner_tags = data_tuple[3]
    
    if "labels" not in tokenized_datasets["train"].features:
        tokenized_datasets = tokenized_datasets.rename_column(args.label_name, "labels")
        
    # if we are doing few shot - we need to sample the training data
    if few_shot_n is not None:
        train_datasets = []
        for label in range(num_labels):
            label_dataset = tokenized_datasets['train'].filter(lambda x: x['labels'] == label).shuffle(seed=42)
            num_samples = len(label_dataset)
            # if we have more samples than the few shot n - then we need to sample
            if num_samples >= few_shot_n:

                # select num_samples_per_class samples from the label
                label_dataset = label_dataset.select(range(few_shot_n))
            
            # add to list of datasets
            train_datasets.append(label_dataset)

        tokenized_datasets["train"] = concatenate_datasets(train_datasets)

    print(f'Sample train data:\n{tokenized_datasets["train"][10]}')
    print(f'\nSample train data (decoded):'+
            f'{tokenizer.decode(tokenized_datasets["train"][10]["input_ids"])}')        
    # print length of datasets
    print(f"tokenized datasets:\n {tokenized_datasets['validation']}")
    print(f"tokenized datasets:\n {tokenized_datasets['train']}")

    ######################### Model setup #########################
    loguru_logger.info("Setting up model")
    
    model_args = dict(pretrained_model_name_or_path=model_name_or_path, 
                          num_labels=num_labels, 
                          output_hidden_states=False, 
                          trust_remote_code=True)
    if args.eight_bit_training:
        loguru_logger.info("Using 8 bit training")
        
            
        # model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path,
        #                                                            torch_dtype=torch.bfloat16,
        #                                                            num_labels = num_labels,return_dict=True)
        # 8 bit
        model_args.update(dict(torch_dtype=torch.float16, 
                            load_in_8bit=True, 
                            device_map="auto"))
        
    if task_type == "SEQ_CLS":
        model = AutoModelForSequenceClassification.from_pretrained(**model_args)
    elif task_type == "TOKEN_CLS":
        model = AutoModelForTokenClassification.from_pretrained(**model_args)
    
    # falcon model seems to use model config to define pad token and the remote code panicks if you don't set it
    if "falcon" in model_name_or_path:
        model.config.use_cache = False
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # need to now prepare the 8bit models
    if args.eight_bit_training:
        fp16_flag = True
        prepare_model_for_kbit_training(model)
    else:#NOTE maybe we want to do fp16 for all anyway
        fp16_flag = False
    
    #########uncomment below for PEFT models#########
    
    # if not pethod method supplied - do full-finetuning
    #TODO  - edit below
    if peft_method == "Full":
        loguru_logger.info("Using full finetuning")
        lr = 3e-5
        peft_config = None
    else:
        # set up some PEFT params
        peft_config, lr = create_peft_config(peft_method, model_name_or_path,task_type)
        model = get_peft_model(model, peft_config)
        print(f"peft config is: {peft_config}")
        # print(model)
        model.print_trainable_parameters()
        
    # send move to device i.e. cuda
    model.to(device)
    
    
    ######################### Trainer setup #########################
    # set up trainer arguments

    # setup optimizer and lr_scheduler
    optimizer = AdamW(params=model.parameters(), lr=lr)    

    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0.06 * (len(tokenized_datasets['train'])/train_batch_size * num_epochs),
        num_training_steps=(len(tokenized_datasets['train'])/train_batch_size * num_epochs),
    )
    
    monitor_metric_name = "f1_macro"
    
    if task_type == "SEQ_CLS":
       compute_metrics = compute_seq_cls_metrics
    else:
       metric = evaluate.load("seqeval")
       compute_metrics = partial(compute_token_cls_metrics, 
                                 label_list=all_ner_tags, 
                                 metric=metric)
    
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
        fp16 = fp16_flag,
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
        data_collator=collate_fn,
        optimizers =(optimizer, lr_scheduler)
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