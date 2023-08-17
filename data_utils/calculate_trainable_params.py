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
from model_utils import count_trainable_parameters
import pandas as pd
from tqdm import tqdm
# import sys and append path
import sys
sys.path.append("../")
from peft_trainer import create_peft_config
import yaml


def get_number_of_trainable_params(model_names:list,
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
        
        for peft_method in tqdm(peft_types, desc=f"model type: {model_name_or_path}"):
            
            if peft_method == "Full":
                peft_model = model
            else:
                # set up some PEFT params
                peft_config, lr = create_peft_config(peft_method, model_name_or_path,task_type)
                peft_model = get_peft_model(model, peft_config)
                print(f"peft config is: {peft_config}")
                model.print_trainable_parameters()
                
            # lets also confirm this directly and save to args
            # The reason base_model is called twice because the
            # get_peft_model adds an additional wrapper arounf the
            # original base model
            n_trainable_params = count_trainable_parameters(peft_model)

            if hasattr(peft_model, 'classifier'):
                n_classifier_params = count_trainable_parameters(peft_model.classifier)
            else:
                n_classifier_params = count_trainable_parameters(peft_model.score)
            n_peft_params = n_trainable_params - n_classifier_params
            
            # proportion of total trainable params
            n_peft_params_perc = (n_peft_params / total_params) * 100
            
            # store the model name, peft method and number of trainable params
            model_dict[peft_method] = {"n_peft_params": n_peft_params,
                                 "total_params": total_params,
                                 "n_peft_params_perc": n_peft_params_perc}
            
        model_peft_dict[model_name_or_path] = model_dict

    return model_peft_dict

    
# now save to yaml
if __name__ == "__main__":
    
    
    model_name_or_path = [
        'michiyasunaga/BioLinkBERT-base',
       'emilyalsentzer/Bio_ClinicalBERT',
       'yikuan8/Clinical-Longformer',
       'michiyasunaga/LinkBERT-base',
       'nlpie/bio-mobilebert',
       'nlpie/distil-biobert', 
       'decapoda-research/llama-7b-hf',
       'meta-llama/Llama-2-7b-hf',
       'ybelkada/falcon-7b-sharded-bf16',
       '/mnt/sdc/niallt/saved_models/language_modelling/mimic/mimic-roberta-base/sampled_250000/22-12-2022--12-45/checkpoint-100000/',
       '/mnt/sdc/niallt/saved_models/declutr/mimic/few_epoch/mimic-roberta-base/2_anch_2_pos_min_1024/transformer_format/',
       'roberta-base',
       '/mnt/sdc/niallt/saved_models/language_modelling/mimic/roberta-base-mimic-note-custom_pretraining_max_epoch_2_weighted/sampled_250000/07-07-2023--08-30/checkpoint-30000/',
       'nlpie/tiny-biobert']


    model_type_mappings = {
                "roberta-base": "roberta-base",
                "bert": "bert-base-uncased",
                "mobile-biobert": "nlpie/bio-mobilebert",
                "distil-biobert": "nlpie/distil-biobert",
                "tiny-biobert": "nlpie/tiny-biobert",
                "llama-7b": "meta-llama/Llama-2-7b-hf",
                }

    peft_types = ["PROMPT_TUNING","LORA", "PREFIX_TUNING", "P_TUNING", "Full"]

    
    trainable_params_dict = get_number_of_trainable_params(
        model_name_or_path, peft_types, task_type="SEQ_CLS", num_labels=2)

    
    # convert to dict and write to yaml
    with open('../trainable_params.yaml', 'w') as f:
        yaml.dump(trainable_params_dict,f, default_flow_style=False)
    

