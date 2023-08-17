import os
from pathlib import Path
import shutil

root_dir = '/mnt/sdd/efficient_ml_data/saved_models/peft'

for root, dirs, files in os.walk(root_dir,):    
    if any(['tfevents' in f for f in files]):
        print(root)
        print(dirs)
        print(files)
        print()
    # break
    # if 'fewshot' in root:
        
    #     for subdir in dirs:
    #         if 'bio-mobile' in subdir:                
    #             path_to_delete = os.path.join(root, subdir)
    #             print(f"Deleting folder: {path_to_delete}")
    #             # shutil.rmtree(path_to_delete)