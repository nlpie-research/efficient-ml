from peft import (LoraConfig, PeftType, PrefixTuningConfig,
                  PromptEncoderConfig, PromptTuningConfig, TaskType,
                  get_peft_config, get_peft_model, get_peft_model_state_dict,
                  prepare_model_for_int8_training,
                  prepare_model_for_kbit_training, set_peft_model_state_dict)
from scipy.special import softmax
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification,
                          AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification,
                          LlamaForSequenceClassification, LlamaTokenizer,
                          Trainer, TrainingArguments,
                          get_linear_schedule_with_warmup, set_seed)
from model_utils import count_trainable_parameters, get_model_size, get_full_model_size
import pandas as pd
from tqdm import tqdm
# import sys and append path
import sys
sys.path.append("../")
from peft_trainer import create_peft_config
import yaml
import copy
import argparse
from thop import profile # for flops calc

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()
    # Required parameters

    parser.add_argument("--log_save_dir",
                        default = "/mnt/sdc/niallt/saved_models/peft_training/logs",
                        type=str,
                        help = "The data path to save tb log files to")

    parser.add_argument("--task_type",
                        default="SEQ_CLS", # SEQ-CLS
                        choices=["SEQ_CLS", "TOKEN_CLS"],
                        type=str,
                        help="name of dataset")

    parser.add_argument("--peft_method",
                        default="LORA", # LORA, PREFIX_TUNING, PROMPT_TUNING, P_TUNING
                        type=str,
                        help="Which peft method to use") 
    parser.add_argument("--lora_rank",
                        type=int,
                        default = 8)
    parser.add_argument("--lora_alpha",
                        type=int,
                        default = 16)
    parser.add_argument("--lora_dropout",
                        type=int,
                        default = 0.1)
    parser.add_argument("--learning_rate",
                        type=float,
                        default = 3e-4)
    parser.add_argument("--num_virtual_tokens",
                        type=int,
                        default = 10)  


    args = parser.parse_args()
    return args

# create args

args = parse_args()

def get_number_of_trainable_params(args:argparse.Namespace,
                                   model_names:list,
                                   peft_types:list,
                                   task_type:str = "SEQ_CLS",
                                   num_labels:int = 2):

    # set up empty dicts to full for dfs
    model_peft_dict = {}
    
    for model_name_or_path in model_names:
        
        model_dict = {}
        # model_name_or_path = model_type_mappings[model_type]
        model_args = dict(pretrained_model_name_or_path=model_name_or_path, 
                          num_labels=num_labels, 
                          output_hidden_states=False, 
                          trust_remote_code=True)

            
        if task_type == "SEQ_CLS":
            model = AutoModelForSequenceClassification.from_pretrained(**model_args)
        elif task_type == "TOKEN_CLS":
            model = AutoModelForTokenClassification.from_pretrained(**model_args)
        
        # falcon model seems to use model config to define pad token and the remote code panicks if you don't set it
        if "falcon" in model_name_or_path:
            model.config.use_cache = False            

        # count total trainable params before peft
        total_params = count_trainable_parameters(model)
        
        # get model size and full model size too
        model_size_MB, model_size_GB = get_model_size(model)
        full_model_size_MB, full_model_size_GB = get_full_model_size(model)
        
        for peft_method in tqdm(peft_types, desc=f"model type: {model_name_or_path}"):
            
            try:
                if peft_method == "Full":
                    peft_model = copy.deepcopy(model)
                    # FLOPs too
                    input_ids = torch.tensor([[101, 2023, 2003, 1037, 2047, 2814, 1012, 102]])                
                    raw_flops, params = profile(peft_model, inputs=(input_ids,))
                    # convert flops to scientific notation
                    flops = "{:.2e}".format(raw_flops)
                else:
                    # hardcode raw_flops and flops for PEFT - it doesn't work with the thop package
                    raw_flops, flops = None, None
                    # set up some PEFT params
                    peft_config, lr = create_peft_config(args, peft_method, model_name_or_path, task_type)
                    peft_model = get_peft_model(copy.deepcopy(model), peft_config)
                    print(f"peft config is: {peft_config}")
                    peft_model.print_trainable_parameters()
                    
                # lets also confirm this directly and save to args
                # The reason base_model is called twice because the
                # get_peft_model adds an additional wrapper arounf the
                # original base model
                n_trainable_params = count_trainable_parameters(peft_model)
                print(f"n_trainable_params: {n_trainable_params}")

                if hasattr(peft_model, 'classifier'):
                    n_classifier_params = count_trainable_parameters(peft_model.classifier)
                    
                else:
                    n_classifier_params = count_trainable_parameters(peft_model.score)
                
                print(f"n_classifier_params: {n_classifier_params}")
                n_peft_params = n_trainable_params - n_classifier_params
                print(f"n_peft_params: {n_peft_params}")
                
                # proportion of total trainable params
                n_peft_params_perc = (n_peft_params / total_params) * 100
                
                # get size of peft adapter only
                peft_model_size_MB, peft_model_size_GB = get_model_size(peft_model)
                peft_full_model_size_MB, peft_full_model_size_GB = get_full_model_size(peft_model)
                
   
                
                # store the model name, peft method and number of trainable params
                # model_dict[peft_method] = {"n_peft_params": n_peft_params,
                #                     "total_params": total_params,
                #                     "n_peft_params_perc": n_peft_params_perc}
                
                # store the model name, peft method and number of trainable params
                model_dict[peft_method] = {"n_peft_params": n_peft_params,
                                    "total_params": total_params,
                                    "n_peft_params_perc": n_peft_params_perc,
                                    "n_trainable_params": n_trainable_params,                             
                                    
                                    "model_size_MB": model_size_MB,
                                    "model_size_GB": model_size_GB,
                                    "full_model_size_MB": full_model_size_MB,
                                    "full_model_size_GB": full_model_size_GB,
                                    "peft_model_size_MB": peft_model_size_MB,
                                    "peft_model_size_GB": peft_model_size_GB,
                                    "peft_full_model_size_MB": peft_full_model_size_MB,
                                    "peft_full_model_size_GB": peft_full_model_size_GB,
                                    "FLOPs": flops,
                                    "raw_FLOPS": raw_flops}
                
            except Exception as e:
                print(f"Error for {model_name_or_path} and {peft_method}")
                print(e)
                model_dict[peft_method] = {"n_peft_params": None,
                                    "total_params": None,
                                    "n_peft_params_perc": None,
                                    "n_trainable_params": None,
                                    "total_trainable_params": None,
                                    
                                    "model_size_MB": None,
                                    "model_size_GB": None,
                                    "full_model_size_MB": None,
                                    "full_model_size_GB": None,
                                    "peft_model_size_MB": None,
                                    "peft_model_size_GB": None,
                                    "peft_full_model_size_MB": None,
                                    "peft_full_model_size_GB": None,
                                    "FLOPs": None,
                                    "raw_FLOPS": None}
                
                
            
        model_peft_dict[model_name_or_path] = model_dict

    return model_peft_dict

    
# now save to yaml
if __name__ == "__main__":
    
    
    model_name_or_path = [
        # 'michiyasunaga/BioLinkBERT-base',
       'emilyalsentzer/Bio_ClinicalBERT',
    #    'yikuan8/Clinical-Longformer',
    #    'michiyasunaga/LinkBERT-base',
       'nlpie/bio-mobilebert',
       'nlpie/distil-biobert', 
       'meta-llama/Llama-2-7b-hf',
    #    '/mnt/sdc/niallt/saved_models/language_modelling/mimic/mimic-roberta-base/sampled_250000/22-12-2022--12-45/checkpoint-100000/',
    #    '/mnt/sdc/niallt/saved_models/declutr/mimic/few_epoch/mimic-roberta-base/2_anch_2_pos_min_1024/transformer_format/',
       'roberta-base',
       'dmis-lab/biobert-v1.1',
    #    '/mnt/sdc/niallt/saved_models/language_modelling/mimic/roberta-base-mimic-note-custom_pretraining_max_epoch_2_weighted/sampled_250000/07-07-2023--08-30/checkpoint-30000/',
       'nlpie/tiny-biobert',
       'bert-base-uncased',
         'google/mobilebert-uncased',
            'prajjwal1/bert-tiny',
            'distilbert-base-uncased',
            'nlpie/bio-distilbert-uncased',            
       'huawei-noah/TinyBERT_General_4L_312D',
       'nlpie/bio-distilbert-uncased', 
       'nlpie/clinical-distilbert',
       'nlpie/clinical-mobilebert',       
     'nlpie/tiny-clinicalbert'
       ]
    
    
    # model_name_or_path = [
    
    #  'nlpie/tiny-clinicalbert'
    #    ]
    


    model_type_mappings = {
                "roberta-base": "roberta-base",
                "bert": "bert-base-uncased",
                "mobile-biobert": "nlpie/bio-mobilebert",
                "distil-biobert": "nlpie/distil-biobert",
                "tiny-biobert": "nlpie/tiny-biobert",
                "llama-7b": "meta-llama/Llama-2-7b-hf",
                }

    peft_types = ["LORA", "PREFIX_TUNING", "IA3", "Full"]

    
    trainable_params_dict = get_number_of_trainable_params(
        args, model_name_or_path, peft_types, task_type="SEQ_CLS", num_labels=2)

    
    # convert to dict and write to yaml
    with open('../trainable_params.yaml', 'w') as f:
        yaml.dump(trainable_params_dict,f, default_flow_style=False)
    

