import os

import matplotlib.pyplot as plt
import optuna
import pandas as pd
import seaborn as sns
import yaml

db_file = 'sqlite:///./Runs/optuna/peft_optuna_v2.db'
study_names = optuna.study.get_all_study_names(storage=db_file)
merged_csv = './Runs/optuna/peft_optuna_v2.csv'

if not os.path.exists(merged_csv):
    merged_df = []
    for study_name in study_names:
        if 'LORARank' not in study_name:
            continue
        study = optuna.load_study(study_name=study_name, 
                                storage=db_file)
        study_df = study.trials_dataframe()
        # print(study_name)
        # print(study_df.head())
        study_df['model_name'] = study_name.split('_')[0]
        merged_df.append(study_df)

    merged_df = pd.concat(merged_df)
    merged_df.rename(columns={'value': 'AUROC_macro'}, inplace=True)
    merged_df.to_csv('./Runs/optuna/peft_optuna_v2.csv', index=False)
else:
    merged_df = pd.read_csv(merged_csv)


# Best params per model
best_params = merged_df.groupby('model_name')['AUROC_macro'].idxmax()
best_params = merged_df.loc[best_params]
best_params = best_params[['model_name', 'params_lora_rank', 'params_lora_alpha', 
                           'params_lora_dropout', 'params_learning_rate', 'AUROC_macro']]
best_params['params_lora_alpha'] = best_params['params_lora_rank']*best_params['params_lora_alpha']
best_params.rename(columns={'params_lora_rank': 'lora_rank', 'params_lora_alpha': 'lora_alpha',
                            'params_lora_dropout': 'lora_dropout', 'params_learning_rate': 'learning_rate'}, 
                            inplace=True) 
best_params.to_csv('./Runs/optuna/best_params_LORA.csv', index=False)

plot_df = merged_df.groupby(['model_name', 'params_lora_rank'])['AUROC_macro'].max().reset_index()
plot_df.sort_values(by=['model_name', 'params_lora_rank'], inplace=True)
plot_df['AUROC_diff'] = plot_df.groupby('model_name')['AUROC_macro'].transform(lambda x: x - x.iloc[0])
plot_df['params_lora_rank'] = plot_df['params_lora_rank'].astype(str)
plot_df.rename(columns={'params_lora_rank': 'LoRA rank', 
                        'AUROC_macro': 'AUROC (macro)'}, inplace=True)

# rename model names for plotting
model_rename_dict = {'bio-mobilebert': 'Mobile-BERT',
                     'biobert-v1.1': 'BERT',
                     'distil-biobert': 'Distil-BERT',
                     'tiny-biobert': 'Tiny-BERT',
                     'roberta-base': 'RoBERTa',}
for k, v in model_rename_dict.items():
    plot_df['model_name'] = plot_df['model_name'].str.replace(k, v)

ax = sns.lineplot(data=plot_df, x='LoRA rank', y='AUROC_diff', 
              hue='model_name', alpha=0.5, errorbar=None, palette='Set2')
ax.set_ylabel('$ AUROC_x - AUROC_8 $')
plt.tight_layout()
plt.savefig('./Runs/optuna/peft_optuna_v2.png', dpi=300)
plot_df.to_csv('./Runs/optuna/peft_optuna_v2_plot_df.csv', index=False)