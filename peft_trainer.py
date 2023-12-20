import argparse
import copy
import json
import os
import sys
import traceback
from datetime import datetime
from functools import partial

import evaluate
import numpy as np
import optuna
import torch
import yaml
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from loguru import logger as loguru_logger
from peft import (
    IA3Config,
    LoraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    TaskType,
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from scipy.special import softmax

# from torchmetrics import AUROC # cuda issues
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchmetrics import AUROC
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    LlamaForSequenceClassification,
    LlamaTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    get_constant_schedule,
    get_linear_schedule_with_warmup,
    set_seed,
)

from data_utils.model_utils import (
    count_trainable_parameters,
    freeze_model,
    unfreeze_model,
)

# print cuda_visible_devices 

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 6,7
# 2080s = 0,3,5,6,8 
# When specifiying from the command line, Nvidia-smi ids: 0, 3, 5, 6, 8 Actual id: 5,6,7,8,9 





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

class TimeBudgetCallback(TrainerCallback):
    def __init__(self, time_limit:int, start_time: datetime):
        self.time_limit = time_limit
        self.start_time = start_time

    def on_step_end(self, args: TrainingArguments, state: TrainerState, 
                    control: TrainerControl, **kwargs):
        elapsed_time = datetime.now() - self.start_time
        if elapsed_time.seconds > self.time_limit:
            control.should_training_stop = True
            print("Time budget exceeded. Stopping training.")
    
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, 
                    control: TrainerControl, **kwargs):
        elapsed_time = datetime.now() - self.start_time
        print(f"Time budget callback: {elapsed_time.seconds} seconds elapsed.")
    

class DatasetInfo:
  def __init__(self, name,
               ds_type="ner", 
               metric=None, 
               load_from_disk=True,
               isMultiSentence=False, 
               validationSubsets=["validation","test"],
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
    parser.add_argument("--gradient_accumulation_steps",
                        default = 1,
                        type=int,
                        help = "the number of gradient accumulation steps")
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
    parser.add_argument('--save_adapter',
                        action = "store_true",
                        help='Whether or not to save the trained adapter weights')

    parser.add_argument("--optimizer",
                        default="adamw",
                        type=str,
                        help="Optimization algorithm to use e.g. adamw, adafactor")

    parser.add_argument("--peft_method",
                        default="LORA", # LORA, PREFIX_TUNING, PROMPT_TUNING, P_TUNING
                        type=str,
                        help="Which peft method to use") 
    parser.add_argument("--lora_rank",
                        type=int,
                        default = 8)
    parser.add_argument("--lora_alpha",
                        type=int,
                        default = 8)
    parser.add_argument("--lora_dropout",
                        type=float,
                        default = 0.1)
    parser.add_argument("--learning_rate",
                        type=float,
                        default = 3e-4)
    parser.add_argument("--num_virtual_tokens",
                        type=int,
                        default = 10)  
    parser.add_argument("--few_shot_n",
                        type=int,
                        default = None)
    parser.add_argument("--eval_few_shot_n",
                        type=int,
                        default = 128)
    parser.add_argument("--optuna",
                        action = "store_true",
                        help='Whether or not to use optuna to tune hyperparameters')
    parser.add_argument('--combined_val_test_sets',
                        default=False,
                        type=bool,
                        help='Whether or not to combine the validation and test datasets')
    parser.add_argument('--unfreeze_all',
                        action = "store_true",
                        help='Whether to unfreeze all layers of the model - even after PEFT')
    parser.add_argument('--no_cuda',
                        action = "store_true",        
                        help='Whether to use cuda/gpu or just use CPU ')
    parser.add_argument('--time_budget',
                        default=-1,
                        type=int,
                        help='Time budget in seconds. If -1, no time budget is used.')
    parser.add_argument('--early_stopping_patience',
                        default=5,
                        type=int,
                        help='Early stopping patience in epochs.')
    parser.add_argument('--early_stopping_threshold',
                        default=0.0,
                        type=float,
                        help='Early stopping threshold.')
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
                                "mimic-los": ("text", None),
                                "mednli": ("sentence1", "sentence2"),
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

    # add few shot sampling here too
    if few_shot_n is not None:
        loguru_logger.info(f"Sampling {few_shot_n} samples per class")
        train_datasets = []
        for label in range(num_labels):
            label_dataset = datasets['train'].filter(lambda x: x[args.label_name] == label).shuffle(seed=42)
            num_samples = len(label_dataset)
            # if we have more samples than the few shot n - then we need to sample
            if num_samples >= few_shot_n:

                # select num_samples_per_class samples from the label
                label_dataset = label_dataset.select(range(few_shot_n))
            
            # add to list of datasets
            train_datasets.append(label_dataset)

        datasets["train"] = concatenate_datasets(train_datasets)
    # same for validation
    if args.eval_few_shot_n is not None:
        loguru_logger.info(f"Sampling {args.eval_few_shot_n} samples per class")
        eval_datasets = []
        for label in range(num_labels):
            label_dataset = datasets['validation'].filter(lambda x: x[args.label_name] == label).shuffle(seed=42)
            num_samples = len(label_dataset)
            # if we have more samples than the few shot n - then we need to sample
            if num_samples >= args.eval_few_shot_n:

                # select num_samples_per_class samples from the label
                label_dataset = label_dataset.select(range(args.eval_few_shot_n))
            
            # add to list of datasets
            eval_datasets.append(label_dataset)

        datasets["validation"] = concatenate_datasets(eval_datasets)

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

def load_datasets(args:argparse.Namespace, info:DatasetInfo, tokenizer:AutoTokenizer) -> tuple:
    """#Dataset Utilities"""
    
    if not info.load_from_disk:
      dataset = load_dataset(info.name)
    else:
      dataset = load_from_disk(info.name)
      
    loguru_logger.info(f"Dataset loaded. Dataset info: {dataset}")

    if info.type == "classification":
        num_labels = len(set(dataset["train"]["labels"]))
      #TODO - add the fewshot sampling here?
        # if we are doing few shot - we need to sample the training data
        if args.few_shot_n is not None:
            loguru_logger.info(f"Sampling {args.few_shot_n} samples per class")
            train_datasets = []
            for label in range(num_labels):
                label_dataset = dataset['train'].filter(lambda x: x['labels'] == label).shuffle(seed=42)
                num_samples = len(label_dataset)
                # if we have more samples than the few shot n - then we need to sample
                if num_samples >= args.few_shot_n:

                    # select num_samples_per_class samples from the label
                    label_dataset = label_dataset.select(range(args.few_shot_n))
                
                # add to list of datasets
                train_datasets.append(label_dataset)

            dataset["train"] = concatenate_datasets(train_datasets)
        # also for eval set using the args.eval_few_shot_n
        if args.eval_few_shot_n is not None:
            loguru_logger.info(f"Sampling {args.eval_few_shot_n} samples per class")
            eval_datasets = []
            for label in range(num_labels):
                label_dataset = dataset['validation'].filter(lambda x: x['labels'] == label).shuffle(seed=42)
                num_samples = len(label_dataset)
                # if we have more samples than the few shot n - then we need to sample
                if num_samples >= args.eval_few_shot_n:

                    # select num_samples_per_class samples from the label
                    label_dataset = label_dataset.select(range(args.eval_few_shot_n))
                
                # add to list of datasets
                eval_datasets.append(label_dataset)

            dataset["validation"] = concatenate_datasets(eval_datasets)

        def mappingFunction(samples, **kargs):
            if info.isMultiSentence:
                outputs = tokenizer(samples[dataset["train"].column_names[0]],
                                    samples[dataset["train"].column_names[1]],
                                    max_length=args.max_length,
                                    truncation=True,
                                    padding=kargs["padding"])
            else:
                outputs = tokenizer(samples[dataset["train"].column_names[0]],
                                    truncation=True,
                                    max_length=args.max_length,
                                    padding=kargs["padding"])

            outputs["labels"] = samples["labels"]

            return outputs
    elif info.type == "ner":
      # print(dataset)
      num_labels = len(dataset["info"][0]["all_ner_tags"])
      loguru_logger.info(f"Number of labels for NER task is: {num_labels}")
      def mappingFunction(all_samples_per_split, **kargs):
        tokenized_samples = tokenizer.batch_encode_plus(all_samples_per_split["tokens"],
                                                        is_split_into_words=True, 
                                                        truncation=True,
                                                        max_length=args.max_length,
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
    # here we want to ensure we get both a validation and a test set
    if "test" not in dataset:
        
        loguru_logger.info("No test set found. Creating test set from validation set")
        temp_dataset = dataset["validation"].train_test_split(test_size=0.5, shuffle=True, seed=42)
        # reassign the test set
        # print(f"temp dataset is: {temp_dataset}")
        dataset["validation"] = temp_dataset["train"]
        dataset["test"] = temp_dataset["test"]  
        # print(f"dataset is: {dataset}")
       
    
    # do not merge the validation and test sets
    if not args.combined_val_test_sets:
        loguru_logger.info("Not combining validation and test sets! Creating test dataset")
        tokenizedValDataset = dataset["validation"].map(mappingFunction,
                                                batched=True,
                                                remove_columns=dataset["validation"].column_names,
                                                fn_kwargs={"padding": "do_not_pad"})
        tokenizedTestDataset = dataset["test"].map(mappingFunction,
                                                batched=True,
                                                remove_columns=dataset["test"].column_names,
                                                fn_kwargs={"padding": "do_not_pad"})
        
    else:
        raise NotImplementedError("Combining validation and test sets not implemented yet")
            
            # tokenizedValDatasets.append(tokenizedValDataset)
            
        # tokenizedValDataset = dataset[name].map(mappingFunction,
        #                                         batched=True,
        #                                         remove_columns=dataset[name].column_names,
        #                                         fn_kwargs={"padding": "do_not_pad"})
        
        # tokenizedValDatasets.append(tokenizedValDataset)

    if info.num_labels != None:
      num_labels = info.num_labels

    if info.type == "ner":
        # check if tokenized test dataset exists
        if "tokenizedTestDataset" in locals():
            print("Returning tokenized test dataset")
            return tokenizedTrainDataset, tokenizedValDataset, tokenizedTestDataset, num_labels, dataset["info"][0]["all_ner_tags"]
        else:
            return tokenizedTrainDataset, tokenizedValDataset, num_labels, dataset["info"][0]["all_ner_tags"]
    else:
        # check if tokenized test dataset exists
        if "tokenizedTestDataset" in locals():
            print("Returning tokenized test dataset")
            return tokenizedTrainDataset, tokenizedValDataset, tokenizedTestDataset, num_labels
        else:
            return tokenizedTrainDataset, tokenizedValDataset, num_labels
        

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
    print(f"logits are: {logits} of shape: {logits.shape}")
    # print(f"logits shape is: {logits.shape}")
    pred_scores = softmax(logits, axis = -1)        
    predictions = np.argmax(logits, axis = -1)
    
    
    # check if pred_scores sum to 1
            
    # print(f"Labels are: {labels}\n")
    # print(f"Preds are: {predictions}")
    # print(f"Pred scores are: {pred_scores}")
    
    # roc_auc_score needs to handle both binary and multiclass
    # check shape of logits to determine which to use
    
    # compute roc_auc using torchmetrics

    
    # if logits.shape[1] == 1 or logits.shape[1] == 2:
    #     auroc = AUROC(task = "binary")
    #     roc_auc = auroc(torch.tensor((pred_scores[:,1])), torch.tensor(labels))
    # else:
    #     auroc = AUROC(task = "multiclass", num_classes = logits.shape[1])
    #     roc_auc = auroc(torch.tensor((pred_scores)), torch.tensor(labels))
        
    # or using evaluate
    # try:
    #     if logits.shape[1] == 1 or logits.shape[1] == 2:
    #         roc_auc_score = evaluate.load("roc_auc", "binary")
    #         roc_auc = roc_auc_score.compute(references=labels,
    #                                         # just take the probabilties of the positive class
    #                                         prediction_scores = pred_scores[:,1]                                         
    #                                         )['roc_auc']
    #     else:
    #         roc_auc_score = evaluate.load("roc_auc", "multiclass")

    #         roc_auc = roc_auc_score.compute(references=labels,
    #                                         prediction_scores = pred_scores,
    #                                         multi_class = 'ovr', 
    #                                         average = "macro")['roc_auc']
    # except:
    #     with open("./faulty_logits.txt", "w") as f:
    #         f.write("\n".join([str(l) for l in logits]))
        
    #     # same for scores
    #     with open("./faulty_scores.txt", "w") as f:
    #         f.write("\n".join([str(l) for l in pred_scores]))
        
    #     # exit here
    #     # exit(0)
    #     # default auroc to 0
    #     roc_auc = 0.0

    # roc_auc using sklearn
    try:
        if logits.shape[1] == 1 or logits.shape[1] == 2:
            roc_auc = roc_auc_score(labels, pred_scores[:,1])
        else:
            roc_auc = roc_auc_score(labels, pred_scores, multi_class = 'ovr', average = "macro")
    except:
        with open("./faulty_logits.txt", "w") as f:
            f.write("\n".join([str(l) for l in logits]))
        
        # same for scores
        with open("./faulty_scores.txt", "w") as f:
            f.write("\n".join([str(l) for l in pred_scores]))
        
        # exit here
        # exit(0)
        # # default auroc to 0
        roc_auc = 0.0
            
        
    # print(f"logits are: {logits} of shape: {logits.shape}")
    
    print(f"Labels are: {labels}\n")
    print(f"Preds are: {predictions}")
    print(f"Pred scores are: {pred_scores}")
    precision = precision_score.compute(predictions=predictions, references=labels, average = "macro")["precision"]
    recall = recall_score.compute(predictions=predictions, references=labels, average = "macro")["recall"]
    accuracy = accuracy_score.compute(predictions=predictions, references=labels)["accuracy"]
    f1_macro = f1_score.compute(predictions=predictions, references=labels, average = "macro")["f1"]
    f1_micro = f1_score.compute(predictions=predictions, references=labels, average = "micro")["f1"]
    f1_weighted = f1_score.compute(predictions=predictions, references=labels, average = "weighted")["f1"]

    
    return {"precision": precision, 
            "recall": recall,
            "accuracy": accuracy,
            "f1_macro":f1_macro,
            "f1_micro":f1_micro,
            "f1_weighted":f1_weighted,
            "roc_auc_macro":roc_auc}

def create_peft_config(args:argparse.Namespace,peft_method:str, model_name_or_path:str, task_type:str) -> tuple:
    
    #TODO - add the target_modules to a config file
    if peft_method == "LORA":
        loguru_logger.info("Using LORA")
        # according to authors for falcon model adding all linear layers is important
        # in fact, add lora_alpha etc to config too. This is a bit of a mess     
        if "falcon" in model_name_or_path.lower():
            loguru_logger.info("Using falcon config")
            lora_alpha = args.lora_alpha
            lora_dropout = args.lora_dropout
            lora_r = args.lora_rank

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
            lr = args.learning_rate
        elif "mobile" in model_name_or_path.lower():
            loguru_logger.info("Using mobile config")
            peft_type = PeftType.LORA
            lr = args.learning_rate # default 3e-4
            peft_config = LoraConfig(task_type=task_type, target_modules=["query", "key", "value"], inference_mode=False, 
                                    r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
            
        elif "longformer" in model_name_or_path.lower():
            loguru_logger.info("Using longformer config")
            peft_type = PeftType.LORA
            lr = args.learning_rate # default 3e-4
            peft_config = LoraConfig(task_type="SEQ_CLS", target_modules=["query","value","key", 
                                                         "query_global", 
                                                         "value_global",
                                                         "key_global"] ,inference_mode=False,
                                     r=args.lora_rank,
                                     lora_alpha=args.lora_alpha,
                                     lora_dropout=args.lora_dropout)
            
            
        else:
            peft_type = PeftType.LORA
            lr = args.learning_rate # default 3e-4
            peft_config = LoraConfig(task_type=task_type, inference_mode=False, 
                                    r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
            
    elif peft_method == "PREFIX_TUNING":
        loguru_logger.info("Using PREFIX_TUNING")
        peft_type = PeftType.PREFIX_TUNING
        peft_config = PrefixTuningConfig(task_type=task_type, 
                                         num_virtual_tokens=args.num_virtual_tokens)
        lr = args.learning_rate # default 1e-2
    elif peft_method == "PROMPT_TUNING":
        loguru_logger.info("Using PROMPT_TUNING")
        # i think we need to set the embedding dimension explicitly for prompt tuning for mobile bert
        if "mobile" in model_name_or_path:
            peft_type = PeftType.PROMPT_TUNING
            peft_config = PromptTuningConfig(task_type=task_type,
                                             token_dim = 128,
                                            num_virtual_tokens=args.num_virtual_tokens)
        else:
            peft_type = PeftType.PROMPT_TUNING
            peft_config = PromptTuningConfig(task_type=task_type, 
                                            num_virtual_tokens=args.num_virtual_tokens)

        lr = args.learning_rate # default 1e-3
    elif peft_method == "P_TUNING":
        loguru_logger.info("Using P_TUNING")
        peft_type = PeftType.P_TUNING
        peft_config = PromptEncoderConfig(task_type=task_type, 
                                          num_virtual_tokens=args.num_virtual_tokens, 
                                          encoder_hidden_size=128)
        lr = args.learning_rate # default 1e-3
        
    elif peft_method == "IA3":
        #TODO refactor any handling of target_modules to be done in the config
        # need to handle specific model types having different target_modules
        if "mobile" in model_name_or_path:
            peft_type = PeftType.IA3
            peft_config = IA3Config(task_type=task_type,
                                    target_modules=["key",
                                                    "value",
                                                    "output.dense"
                                                    ], 
                                    feedforward_modules=['output.dense'], 
                                    inference_mode=False)
        elif "longformer" in model_name_or_path:            
            peft_type = PeftType.IA3
            peft_config = IA3Config(task_type=task_type,
                                    target_modules=["query","value","key", 
                                                         "query_global", 
                                                         "value_global",
                                                         "key_global", 
                                                         "output.dense"],                                                   
                                    feedforward_modules=['output.dense'], 
                                    inference_mode=False)
        
        else:
            peft_type = PeftType.IA3
            peft_config = IA3Config(task_type=task_type, inference_mode=False)
        
        lr = args.learning_rate # default 1e-3
       
    else:
        raise NotImplementedError(f"peft method: {peft_method} not implemented yet")

    return peft_config, lr

def tune_hyperparams(model, args:argparse.Namespace, trainer:Trainer) -> None:
    
    if not (args.peft_method == "LORA" or args.peft_method == "IA3"):
        loguru_logger.info(f"Optuna only implemented for LORA or IA3 at the moment. But got:{args.peft_method} Exiting.")
        return

    def optuna_hp_space(trial):
        return {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        }

    def optuna_model_init_lora(trial):
        assert args.peft_method == "LORA", "Optuna only implemented for LORA at the moment. Exiting."
        # Set the parameter space anyway to avoid issues while saving/loading
        lora_rank = trial.suggest_categorical('lora_rank', [args.lora_rank])
        lora_alpha = lora_rank * trial.suggest_categorical('lora_alpha', [0.3, 0.5, 1.0])
        lora_dropout = trial.suggest_categorical('lora_dropout', [0.1, 0.3, 0.5])

        args.learning_rate = trial.params['learning_rate']
        args.lora_dropout = lora_dropout
        args.lora_rank = int(lora_rank)
        args.lora_alpha = int(np.round(lora_alpha))

        peft_config, _ = create_peft_config(args=args, 
                                        peft_method=args.peft_method, 
                                        model_name_or_path=args.model_name_or_path, 
                                        task_type=args.task_type)
        loguru_logger.info(f"Optuna params are: {trial.params}")
        
        model_copy = copy.deepcopy(model)
        peft_model = get_peft_model(model_copy, peft_config)
        return peft_model
    
    def optuna_model_init_ia3(trial):
        assert args.peft_method == "IA3", "Optuna only implemented for IA3 at the moment. Exiting."
        
        args.learning_rate = trial.params['learning_rate']
        
        peft_config, _ = create_peft_config(args=args,
                                            peft_method=args.peft_method,
                                            model_name_or_path=args.model_name_or_path,
                                            task_type=args.task_type)
        loguru_logger.info(f"Optuna params are: {trial.params}")
        
        model_copy = copy.deepcopy(model)
        peft_model = get_peft_model(model_copy, peft_config)
        return peft_model
        
    def optuna_objective(metrics):
        return metrics['eval_roc_auc_macro']

    # set study name based on peft_type 
    if args.peft_method == "LORA":
        
        m = args.model_name_or_path.split("/")[-1]
        study_name = f'{m}_LORARank-{args.lora_rank}'
        storage_name = "sqlite:///./Runs/optuna/peft_optuna_v2.db"
    elif args.peft_method == "IA3":
        m = args.model_name_or_path.split("/")[-1]
        study_name = f'{m}_IA3'
        storage_name = "sqlite:////mnt/sdd/efficient_ml_data/optuna_dbs/Runs/peft_optuna_v2.db"
    else:
        raise NotImplementedError(f"Optuna not implemented for {args.peft_method} yet")
    
    # pruner = optuna.pruners.MedianPruner(
    #                     n_startup_trials=10, 
    #                     n_warmup_steps=5)
    
    # setup uptuna based on IA3 or LoRA
    if args.peft_method == "LORA":
        trainer.model_init = optuna_model_init_lora
    elif args.peft_method == "IA3":
        trainer.model_init = optuna_model_init_ia3        
    trainer.hyperparameter_search(
                                direction="maximize",
                                backend="optuna",
                                n_trials=20,
                                study_name=study_name,
                                storage=storage_name,
                                load_if_exists=True,
                                hp_space=optuna_hp_space,
                                compute_objective=optuna_objective)
    
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
    optuna = args.optuna
    time_budget = args.time_budget
    
    # set up some variables to add to checkpoint and logs filenames
    time_now = str(datetime.now().strftime("%d-%m-%Y--%H-%M"))
    
    # define some model specific params for logs etc - this is mainly for custom local models
    # TODO clean this up/improve
    # THIS IS ALL VERY CRUDE AND DEPENDENT ON HAVING TRAINED USING THE SCRIPTS INSIDE THIS REPO - 
    # forward slashes really matter for the naming convention make sure to append the path with a forward slash
    model_name = get_model_name(model_name_or_path)
    
    # save to the args
    args.custom_model_name = model_name
    
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
        if "roberta" in model_name_or_path or "declutr"in model_name_or_path:
            tokenizer_args["add_prefix_space"] = True
            
    if "llama" in model_name_or_path:
        loguru_logger.info("Using llama tokenizer")
        tokenizer = LlamaTokenizer.from_pretrained(**tokenizer_args)
    else:
        tokenizer = AutoTokenizer.from_pretrained(**tokenizer_args)   

    if getattr(tokenizer, "pad_token_id") is None:
        loguru_logger.info(f"Adding pad token manually! Setting pad token to eos token: {tokenizer.eos_token_id}")
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
            ds_info = DatasetInfo(name=data_dir, 
                        metric="f1", load_from_disk=True,
                        ds_type=ds_type, isMultiSentence=False,
                        lr=[5e-5, 2e-5, 1e-5], epochs=3,
                        batch_size=[train_batch_size],
                        runs=1)        
            data_tuple = load_datasets(args = args, info=ds_info, tokenizer=tokenizer)
            # if we have test set i.e. an extra element in tuple, change the order
            if len(data_tuple) == 4:
                tokenized_datasets = DatasetDict()
                tokenized_datasets["train"] = data_tuple[0]
                tokenized_datasets["validation"] = data_tuple[1]
                tokenized_datasets["test"] = data_tuple[2]
                num_labels = data_tuple[3]
            else:
                tokenized_datasets = DatasetDict()
                tokenized_datasets["train"] = data_tuple[0]
                tokenized_datasets["validation"] = data_tuple[1]
                num_labels = data_tuple[2]
        else:
            ds_type = "ner"        
            ds_info = DatasetInfo(name=data_dir, 
                        metric="f1", load_from_disk=True,
                        ds_type=ds_type, isMultiSentence=False,
                        lr=[5e-5, 2e-5, 1e-5], epochs=3,
                        batch_size=[train_batch_size],
                        runs=1)        
            data_tuple = load_datasets(args=args, info=ds_info, tokenizer=tokenizer)
            # check if we have a test set
            if len(data_tuple) == 5:
                tokenized_datasets = DatasetDict()
                tokenized_datasets["train"] = data_tuple[0]
                tokenized_datasets["validation"] = data_tuple[1]
                tokenized_datasets["test"] = data_tuple[2]
                num_labels = data_tuple[3]
                all_ner_tags = data_tuple[4]
            else:
                tokenized_datasets = DatasetDict()
                tokenized_datasets["train"] = data_tuple[0]
                tokenized_datasets["validation"] = data_tuple[1]
                num_labels = data_tuple[2]
                all_ner_tags = data_tuple[3]


    
    if "labels" not in tokenized_datasets["train"].features:
        tokenized_datasets = tokenized_datasets.rename_column(args.label_name, "labels")
    
    # print number of labels
    loguru_logger.info(f"##### Number of labels is: {num_labels} #####")
    
    # # if we are doing few shot - we need to sample the training data
    # if few_shot_n is not None:
    #     loguru_logger.info(f"Sampling {few_shot_n} samples per class")
    #     train_datasets = []
    #     for label in range(num_labels):
    #         label_dataset = tokenized_datasets['train'].filter(lambda x: x['labels'] == label).shuffle(seed=42)
    #         num_samples = len(label_dataset)
    #         # if we have more samples than the few shot n - then we need to sample
    #         if num_samples >= few_shot_n:

    #             # select num_samples_per_class samples from the label
    #             label_dataset = label_dataset.select(range(few_shot_n))
            
    #         # add to list of datasets
    #         train_datasets.append(label_dataset)

    #     tokenized_datasets["train"] = concatenate_datasets(train_datasets)

    print(f"Tokenized datasets: {tokenized_datasets}")
    print(f'Sample train data:\n{tokenized_datasets["train"][1]}')
    print(f'\nSample train data (decoded):'+
            f'{tokenizer.decode(tokenized_datasets["train"][1]["input_ids"])}')        
    # print length of datasets
    print(f"Final tokenized train dataset:\n {tokenized_datasets['train']}")
    print(f"tokenized validation dataset:\n {tokenized_datasets['validation']}")
    # if we have separate test set 
    if "test" in tokenized_datasets:
        print(f"tokenized evaluation dataset:\n {tokenized_datasets['test']}")

    

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
    
    # need to load in fp16 for llama models anyway
    elif "falcon" in model_name_or_path or "llama" in model_name_or_path:
        model_args.update(dict(torch_dtype=torch.bfloat16,                            
                            device_map="auto"))  
        
        
    if task_type == "SEQ_CLS":
        loguru_logger.info("Using sequence classification")
        model = AutoModelForSequenceClassification.from_pretrained(**model_args)
        loguru_logger.info(f"Model is: {model}")
    elif task_type == "TOKEN_CLS":
        model = AutoModelForTokenClassification.from_pretrained(**model_args)
    
    # falcon model seems to use model config to define pad token and the remote code panicks if you don't set it
    if "falcon" in model_name_or_path or "llama" in model_name_or_path:
        loguru_logger.info("Setting pad token manually for falcon/llama model in the model config")
        model.config.use_cache = False
        model.config.pad_token_id = tokenizer.eos_token_id

        fp16_flag = False
        
    
    # need to now prepare the 8bit models
    if args.eight_bit_training:
        fp16_flag = True
        prepare_model_for_kbit_training(model)
    else:#NOTE maybe we want to do fp16 for all anyway
        fp16_flag = False
    
    #########uncomment below for PEFT models#########
    
    # if not pethod method supplied - do full-finetuning
    #TODO  - edit below
    
    # If we are using optuna, do not convert model to the peft version
    # here
    if peft_method == "Full":
        loguru_logger.info("Using full finetuning")
        lr = 3e-5
        peft_config = None
    elif peft_method == "Frozen_PLM":
        loguru_logger.info("Using frozen PLM")
        lr = 3e-5
        peft_config = None
        # freeze the base model
        freeze_model(model.base_model)
        print(f"Number of trainable params: {count_trainable_parameters(model)}")
        args.n_trainable_params = count_trainable_parameters(model)
    else:
        # set up some PEFT params
        peft_config, lr = create_peft_config(args=args, peft_method=peft_method, 
                                            model_name_or_path=model_name_or_path, 
                                            task_type=task_type)
        print(f"peft config is: {peft_config}")
    
    # if we are using peft and not running optuna, get peft model
    if not optuna and peft_method != "Full":

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    # lets also confirm this directly and save to args
    args.n_trainable_params = count_trainable_parameters(model)
    
    # if we want to unfreeze all layers - do so here
    if args.unfreeze_all:
        #NOTE - right now we manually set the logs dir to a different folder
        loguru_logger.warning("Unfreezing all layers!!!")
        unfreeze_model(model)
        # log trainable params now
        model.print_trainable_parameters()
        
    
    # send move to device i.e. cuda
    model.to(device)
    
    ######################### Trainer setup #########################
    
    # add constant scheduler
    # lr_scheduler = get_constant_schedule(optimizer=optimizer)
    
    
    
    if task_type == "SEQ_CLS":
       compute_metrics = compute_seq_cls_metrics
       monitor_metric_name = "f1_macro"
    else:
       metric = evaluate.load("seqeval")
       compute_metrics = partial(compute_token_cls_metrics, 
                                 label_list=all_ner_tags, 
                                 metric=metric)
       monitor_metric_name = "f1"
    
    # setup optimizer and lr_scheduler
    # optimizer = AdamW(params=model.parameters(), lr=lr)

    # Instantiate scheduler
    # lr_scheduler = get_linear_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=0.06 * (len(tokenized_datasets['train'])/train_batch_size * num_epochs),
    #     num_training_steps=(len(tokenized_datasets['train'])/train_batch_size * num_epochs),
    # )
    loguru_logger.warning(f"!!!!!!!!!!Prior to training fp16 flag is: {fp16_flag}!!!!!!!!!!!")
    train_args = TrainingArguments(
        output_dir = f"{ckpt_dir}/",
        evaluation_strategy = args.evaluation_strategy,
        eval_steps = args.eval_every_steps,
        logging_steps = args.log_every_steps,
        logging_first_step = True,    
        save_strategy = 'no' if optuna else args.saving_strategy,
        save_steps = args.save_every_steps,        
        per_device_train_batch_size = train_batch_size,
        per_device_eval_batch_size = eval_batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        load_best_model_at_end=not optuna if args.saving_strategy != "no" else False,#fix this - we still want to be able to decide regardless of the optuna param
        metric_for_best_model=monitor_metric_name,
        push_to_hub=False,
        logging_dir = f"{logging_dir}/",
        save_total_limit=2,
        report_to = 'tensorboard',        
        overwrite_output_dir=True,
        fp16 = fp16_flag,
        no_cuda = args.no_cuda, # for cpu only
        lr_scheduler_type = 'constant_with_warmup' if time_budget != -1 else 'linear',
        warmup_steps = 0.06 * (len(tokenized_datasets['train'])/train_batch_size * min(num_epochs, 5)),
        learning_rate = lr,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        # use_ipex = args.use_ipex # for cpu only
        # remove_unused_columns=False, # at moment the peft model changes the output format and this causes issues with the trainer
        # label_names = ["labels"],#FIXME - this is a hack to get around the fact that the peft model changes the output format and this causes issues with the trainer
    )
    
    callbacks = []
    if time_budget != -1:
        time_budget = TimeBudgetCallback(time_limit=time_budget, 
                                            start_time=datetime.now())
        callbacks.append(time_budget)
    
    # setup normal trainer
    trainer = Trainer(
        model,
        train_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,         
        data_collator=collate_fn,
        # optimizers =(optimizer, lr_scheduler),
        callbacks=callbacks
        )

    # If not using optuna, run normal training and logging.
    if not optuna:
        # if the saving strategty is "no" then we do not want early stopping - as this requires saving checkpoints
        if args.saving_strategy != "no":
            loguru_logger.info("Adding early stopping callback")
            early_stopping = EarlyStoppingCallback(
                            early_stopping_patience=args.early_stopping_patience, 
                            early_stopping_threshold=args.early_stopping_threshold)
            trainer.add_callback(early_stopping)

        # run training
        trainer.train()
    
        # save the args/params to a text/yaml file
        with open(f'{logging_dir}/config.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
            
        with open(f'{logging_dir}/config.yaml', 'w') as f:
            yaml.dump(args.__dict__, f) 
        # also save trainer args
        with open(f'{logging_dir}/all_trainer_args.yaml', 'w') as f:
            yaml.dump(trainer.args.__dict__, f)   
            
        
        # save the peft weights to a file
        if args.save_adapter:
            loguru_logger.info(f"Saving adapter weights to: {ckpt_dir}")
            model.save_pretrained(f"{ckpt_dir}")
        
        # run evaluation on test set
        # remove early stopping callbacks here to avoid warnings if we have them - need to check if we did add it
        if args.saving_strategy != "no":
            trainer.remove_callback(early_stopping)
            
        trainer.evaluate(eval_dataset=tokenized_datasets["test"], 
                         metric_key_prefix="test")
        # trainer.predict(test_dataset=tokenized_datasets["test"], 
        #             metric_key_prefix="test")
    else:
        tune_hyperparams(model, args, trainer)

# run script
if __name__ == "__main__":
    
    try:
        main()
    except:
        loguru_logger.warning("Error in main function. Saving error log to file")
        cmd = sys.argv
        m = cmd[cmd.index('--model_name_or_path')+1].split('/')[-1]
        t = cmd[cmd.index('--task')+1]
        p = cmd[cmd.index('--peft_method')+1]
        if '--few_shot_n' in cmd:
            f = cmd[cmd.index('--few_shot_n')+1]
        else:
            f = 'full'
        error_log_file = f'./Runs/{m}_fewshot{f}_{t}_{p}.err'
        with open(error_log_file, 'w') as errf:
            errf.write(traceback.format_exc())


