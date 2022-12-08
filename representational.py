# >>>>>>>>>>>>>>>>>> IMPORTANT <<<<<<<<<<<<<<<<<<
# Comment out or change the "switch_cache" import line, it's only here for me(Brandon)
# because I want my huggingface models to save on a different drive.
import os
import switch_cache  # <----- Comment out
from utils import get_model_names_and_data, finetune
from minicons import scorer
from minicons import cwe
import torch

incremental_models, masked_models, _ = get_model_names_and_data()
profession_file_names = ["Barber_Zarpie_Gen.txt",  "Breeder_Zarpies_Gen.txt",  "Janitor_Zarpies_Gen.txt",  "Nurse_Zarpie_Gen.txt",
                         "Barber_Zarpie_Spe.txt",  "Breeder_Zarpies_Spe.txt",  "Janitor_Zarpies_Spe.txt",  "Nurse_Zarpie_Spe.txt"]

dir_path = os.path.dirname(os.path.realpath(__file__))

context_words = []
professions = []
for dn_ind, data_name in enumerate(profession_file_names):
    data_path = os.path.join(dir_path, 'data', 'professions', data_name)
    p_reader = open(data_path, "r")
    ctx = ""
    for s in p_reader:
        ctx += s.strip()
    context_words.append((ctx, "Zarpies"))
    profession = data_name.split("_")[0]
    professions.append((f'The {profession}s went to store.', f'{profession}s'))
