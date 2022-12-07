# >>>>>>>>>>>>>>>>>> IMPORTANT <<<<<<<<<<<<<<<<<<
# Comment out or change the "switch_cache" import line, it's only here for me(Brandon)
# because I want my huggingface models to save on a different drive.
import os
import switch_cache  # <----- Comment out
from utils import finetune, get_model_names_and_data
import torch

dir_path = os.path.dirname(os.path.realpath(__file__))

# >>>>>>>>>>>>>>>>>> IMPORTANT <<<<<<<<<<<<<<<<<<
# Set this variable to false after you run this code once, having it set to true
# will redo all the finetune training
do_training = False

# The following loop finetunes a model for each combination of unique model + datafile
# This loop will also create a unique name for each model folder
incremental_models, masked_models, finetune_data = get_model_names_and_data()

ignore_list = []
model_paths = []
for dp_ind, data_name in enumerate(finetune_data):
    for inc_ind, model_name in enumerate(incremental_models):
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary(device="cuda", abbreviated=False))
        model_name_clean = model_name.replace("/", "_")
        data_name_clean = data_name.replace(".txt", "")
        out_name = f"{model_name_clean}_adapted_{data_name_clean}"
        if (out_name in ignore_list):
            continue
        out_path = os.path.join(dir_path, 'models', 'incremental', out_name)
        model_paths.append(out_path)
        if (do_training):
            data_path = os.path.join(dir_path, 'data', data_name)
            finetune(model_name, data_path,
                     out_path, use_original=True, masked=False)

    for mas_ind, model_name in enumerate(masked_models):
        torch.cuda.empty_cache()
        # torch.cuda.memory_summary(device="cuda", abbreviated=False)
        model_name_clean = model_name.replace("/", "_")
        data_name_clean = data_name.replace(".txt", "")
        out_name = f"{model_name_clean}_adapted_{data_name_clean}"
        if (out_name in ignore_list):
            continue
        out_path = os.path.join(dir_path, 'models', 'masked', out_name)
        model_paths.append(out_path)
        if (do_training):
            data_path = os.path.join(dir_path, 'data', data_name)
            finetune(model_name, data_path,
                     out_path, use_original=True, masked=True)

print(model_paths)
