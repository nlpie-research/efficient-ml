import os

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import yaml

optuna_dir = '/mnt/sdd/efficient_ml_data/optuna_dbs/Runs'
plots_dir = '/mnt/sdd/efficient_ml_data/figures_and_plots/'

db_file = f'sqlite:///{optuna_dir}/peft_optuna_v2.db'
study_names = optuna.study.get_all_study_names(storage=db_file)
merged_csv = f'{optuna_dir}/peft_optuna_v2.csv'

if not os.path.exists(merged_csv):
    merged_df = []
    for study_name in study_names:
        if 'LORARank' not in study_name:
            continue
        
        model_name = study_name.split('_LORA')[0]
        
        if model_name in ['roberta-base', 'distil-biobert']:
            continue
        if model_name == 'roberta-base_v2':
            model_name = 'roberta-base'
        
        study = optuna.load_study(study_name=study_name, 
                                storage=db_file)
        study_df = study.trials_dataframe()
        study_df['model_name'] = model_name
        merged_df.append(study_df)

    merged_df = pd.concat(merged_df, ignore_index=True)
    merged_df.rename(columns={'value': 'AUROC_macro'}, inplace=True)
    merged_df.to_csv(f'{optuna_dir}/peft_optuna_v2.csv', index=False)
else:
    merged_df = pd.read_csv(merged_csv)

# Best params per model
best_params = merged_df.groupby('model_name')['AUROC_macro'].idxmax()
best_params = merged_df.loc[best_params]
best_params = best_params[['model_name', 'params_lora_rank', 'params_lora_alpha', 
                           'params_lora_dropout', 'params_learning_rate', 'AUROC_macro']]
best_params['params_lora_alpha'] = np.round(best_params['params_lora_rank']*best_params['params_lora_alpha'])
best_params.rename(columns={'params_lora_rank': 'lora_rank', 'params_lora_alpha': 'lora_alpha',
                            'params_lora_dropout': 'lora_dropout', 'params_learning_rate': 'learning_rate'}, 
                            inplace=True) 
best_params['lora_rank'] = best_params['lora_rank'].astype(int)
best_params['lora_alpha'] = best_params['lora_alpha'].astype(int)
print(best_params)
best_params.to_csv(f'{optuna_dir}/best_params_LORA.csv', index=False)

plot_df = merged_df.groupby(['model_name', 'params_lora_rank'])['AUROC_macro'].max().reset_index()
plot_df.sort_values(by=['model_name', 'params_lora_rank'], inplace=True)
plot_df['AUROC_diff'] = plot_df.groupby('model_name')['AUROC_macro'].transform(lambda x: x - x.iloc[0])
plot_df['params_lora_rank'] = plot_df['params_lora_rank'].astype(str)
plot_df.rename(columns={'params_lora_rank': 'LoRA rank', 
                        'AUROC_macro': 'AUROC (macro)'}, inplace=True)
model_rename_dict = {'biobert-v1.1': 'BioBERT',
                     'roberta-base': 'RoBERTa-base',
                     'bio-distilbert-uncased': 'bio-distilbert'}
for k, v in model_rename_dict.items():
    plot_df['model_name'] = plot_df['model_name'].str.replace(k, v)

# rename model names for plotting
HUE_ORDER = ['tiny-biobert', 'bio-mobilebert', 'bio-distilbert', 'BioBERT', 
             'RoBERTa-base']  
plot_df = plot_df.loc[plot_df['model_name'].isin(HUE_ORDER)]

ax = sns.lineplot(data=plot_df, x='LoRA rank', y='AUROC_diff', 
              hue='model_name', hue_order=HUE_ORDER, 
              alpha=0.5, errorbar=None, palette='Set2')
ax.set_ylabel('$ AUROC_x - AUROC_8 $')
plt.tight_layout()
plt.savefig(f'{plots_dir}/peft_optuna_v2.png', dpi=300)
plt.savefig(f'peft_optuna_v2.png', dpi=300)
plot_df.to_csv(f'{optuna_dir}/peft_optuna_v2_plot_df.csv', index=False)