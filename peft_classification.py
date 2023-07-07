import argparse
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from typing import Union
import yaml
import evaluate
import numpy as np
import peft
import torch
from datasets import load_dataset, load_from_disk, load_metric
from loguru import logger as loguru_logger
from peft import (LoraConfig, PeftType, PrefixTuningConfig,
                  PromptEncoderConfig, PromptTuningConfig, TaskType,
                  get_peft_config, get_peft_model, get_peft_model_state_dict,
                  set_peft_model_state_dict)
from seqeval.metrics import f1_score, precision_score, recall_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModel, AutoModelForSequenceClassification,
                          AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, Trainer,
                          TrainingArguments, get_linear_schedule_with_warmup,
                          set_seed)
from functools import partial
import json

class DatasetInfo:
  def __init__(self, name,
               type="ner", 
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
    self.type = type
    self.num_labels = num_labels

    if metric == None:
      self.metric = "accuracy"
    else:
      self.metric = metric

    self.fullName = name + "-" + self.metric

class ModelInfo:
  def __init__(self, pretrainedPath, modelPath, isCustom=False, isAdapterTuning=False, use_token_type_ids=True):
    self.pretrainedPath = pretrainedPath
    self.modelPath = modelPath

    self.logsPath = pretrainedPath + f"/"

    self.isCustom = isCustom
    self.isAdapterTuning = isAdapterTuning
    self.use_token_type_ids = use_token_type_ids

  def get_logs_path(self, datasetName):
    return self.logsPath + f"{datasetName}.txt" if not self.isAdapterTuning else self.logsPath + f"{datasetName}-adapter.txt"
  
  def load_model(self, num_labels, ds):
    if self.isCustom:
      if ds.type == "classification":
        model = AutoModelForSequenceClassification.from_pretrained(self.modelPath, num_labels=num_labels)
      elif ds.type == "ner":
        model = AutoModelForTokenClassification.from_pretrained(self.modelPath, num_labels=num_labels)

      if self.isAdapterTuning:
        model.trainAdaptersOnly()
    else:
      if ds.type == "classification":
        model = AutoModelForSequenceClassification.from_pretrained(self.modelPath, num_labels=num_labels)
      elif ds.type == "ner":
        model = AutoModelForTokenClassification.from_pretrained(self.modelPath, num_labels=num_labels)
    
    return model

class PEFTTrainer:
  def __init__(self) -> None:
    with open("configs.yaml", "r") as f:
      config = yaml.load(f, yaml.FullLoader)
    self.config = config
    self.task_config = None
    
  def set_task_config(self, task_config) -> None:
    self.task_config = task_config
  
  @staticmethod
  def load_datasets(info:DatasetInfo, tokenizer:AutoTokenizer):
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

  def get_datasets(self, ds_info:Union[DatasetInfo, dict], tokenizer:AutoTokenizer, 
                 subset:bool) -> tuple:
  
    data_tuple = None
    if isinstance(ds_info, DatasetInfo):
      train_dataset, valid_dataset, num_labels, all_ner_tags = self.load_datasets(ds_info, tokenizer)
      eval_dataset = valid_dataset[0]
      if subset:
        train_dataset = train_dataset.select(range(0, 3000))
        eval_dataset = eval_dataset.select(range(0, 1000))
    
      data_tuple = (train_dataset, valid_dataset, num_labels, all_ner_tags)

      print(f'Dataset info:')
      print(f'\tMetric: {ds_info.metric}')    
      print(f'\tNER tags: {", ".join(all_ner_tags)}')
      print(f'\tNum labels: {num_labels}')
    
    else:
      datasets = load_dataset("csv", 
                    data_files = {"train":ds_info["path"]["train"],
                                  "validation":ds_info["path"]["val"],
                                  "test":ds_info["path"]["test"]},
                    cache_dir=ds_info["cache_dir"])
      num_labels = len(np.unique(datasets["train"]["label"]))
      sentence1_key, sentence2_key = self.config['TASK2KEY'][self.task_config["task"]]
      
      def tokenize_function(examples):
          # max_length is important when using prompt tuning  or prefix tuning 
          # or p tuning as virtual tokens are added - which can overshoot the 
          # max length in pefts current form
          # for now set to 480 and see how it goes
          if sentence2_key is None:
              return tokenizer(examples[sentence1_key], 
                               truncation=True, 
                               max_length = self.task_config["max_length"])
          return tokenizer(examples[sentence1_key], examples[sentence2_key], 
                          truncation=True, 
                          max_length=self.task_config["max_length"])
      
      tokenized_datasets = datasets.map(
          tokenize_function,
          batched=True,
          remove_columns=['text', 'triage-category'],
      )

      tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
      data_tuple = (
        tokenized_datasets["train"], 
        tokenized_datasets["val"], 
        tokenized_datasets["test"], 
        num_labels)

    return data_tuple

  def compute_metrics(self, p, label_list, metric):
      predictions, labels = p
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

  def run_task(self, peft_config:peft.PeftConfig, model_name:str, dataset_path:str, 
                    logs_path:str, subset:bool=True, batch_size:int=32) -> None:

    model_name = self.task_config["model_name_or_path"]
    
    if model_name == "roberta-base" or "roberta" in model_name:
      tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    else:
      tokenizer = AutoTokenizer.from_pretrained(model_name)

    print('Creating datasets')
    print('======================================')

    #Use the pre-processing code in BioLM
    #(https://github.com/facebookresearch/bio-lm)
    ds_info = DatasetInfo(dataset_path, 
                  metric="f1",
                  load_from_disk=True,
                  type="ner",
                  isMultiSentence=False,
                  lr=[5e-5, 2e-5, 1e-5],
                  epochs=3,
                  batch_size=[batch_size],
                  runs=1)
    datasets = self.get_datasets(ds_info=ds_info, tokenizer=tokenizer, subset=subset)
    train_dataset, eval_dataset, num_labels, all_ner_tags = datasets
    print(f'\nSample train data (decoded): {tokenizer.decode(train_dataset[30]["input_ids"])}')
    print(f'Sample train data:\n{train_dataset[30]}')
    collate_fn = DataCollatorForTokenClassification(tokenizer)

    print('\nLoading and creating PEFT model')
    print('======================================')
    
    model = AutoModelForTokenClassification.from_pretrained(model_name,
                                                        num_labels=num_labels,
                                                        return_dict=True)
    model = get_peft_model(model, peft_config)
    print(f'\nModel parameters:')
    model.print_trainable_parameters()
    
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    model.to(device)
    num_epochs = ds_info.epochs
    metric = evaluate.load("seqeval")
    
    optimizer = AdamW(params=model.parameters(), lr=0.001)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0.06 * (len(datasets['train'])/batch_size * num_epochs),
        num_training_steps=(len(datasets['train'])/batch_size * num_epochs),
    )

    monitor_metric_name = "f1_macro"
    train_args = TrainingArguments(
        output_dir = f"{ckpt_dir}/",
        evaluation_strategy = self.task_config["evaluation_strategy"],
        eval_steps = self.task_config["eval_every_steps"],
        logging_steps = self.task_config["log_every_steps"],
        logging_first_step = True,    
        save_strategy = self.task_config["saving_strategy"],
        save_steps = self.task_config["save_every_steps"],        
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        num_train_epochs=self.task_config["max_epochs"],
        weight_decay=0.01,
        load_best_model_at_end=False,
        metric_for_best_model=monitor_metric_name,
        push_to_hub=False,
        logging_dir = f"{logging_dir}/",
        save_total_limit=2,
        report_to = 'tensorboard',        
        overwrite_output_dir=True
    )
    
    # setup normal trainer
    trainer = Trainer(
        model,
        train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=partial(self.compute_metrics, metric=metric, 
                                label_list=all_ner_tags),
        data_collator= collate_fn,
        optimizers = (optimizer, lr_scheduler)
    )

    print('\nStarting training and evaluation')
    print('======================================')
    trainer.train()

if __name__=='__main__':
  
  with open("configs.yaml", "r") as f:
    config = yaml.load(f, yaml.FullLoader)

  datasets = config["DATASETS"]
  
  for model_name in ["roberta-base"]:
    for peft_config in config["PEFT_CONFIGS"]:
      for ds_name, ds_info in datasets.items():
        logs_path = f"/mnt/sdd/efficient_ml_data/saved_models/peft/{ds_name}/{model_name}/"
        peft_config["task_type"] = ds_info["task_type"]
        
        print('\n===========================================')
        print(f'Model name: {model_name}')
        print(f'Dataset: {ds_name}')
        print('===========================================\n')

        peft_training(peft_config=peft_config, model_name=model_name, 
                      dataset_path=ds_info["path"], logs_path=logs_path, 
                      subset=False, batch_size=16)
      