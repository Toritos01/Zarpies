# >>>>>>>>>>>>>>>>>> IMPORTANT <<<<<<<<<<<<<<<<<<
# Comment out or change the "switch_cache" import line, it's only here for me(Brandon)
# because I want my huggingface models to save on a different drive.
import os
import switch_cache  # <----- Comment out
from utils import get_model_names_and_data
from minicons import scorer
from minicons import cwe
import torch


incremental_models, masked_models, _ = get_model_names_and_data()
profession_file_names = ["Barber_Zarpie_Gen.txt",  "Breeder_Zarpies_Gen.txt",  "Janitor_Zarpies_Gen.txt",  "Nurse_Zarpie_Gen.txt",
                         "Barber_Zarpie_Spe.txt",  "Breeder_Zarpies_Spe.txt",  "Janitor_Zarpies_Spe.txt",  "Nurse_Zarpie_Spe.txt"]

dir_path = os.path.dirname(os.path.realpath(__file__))

context_words = []
professions = []
for dp_ind, data_name in enumerate(profession_file_names):
    data_path = os.path.join(dir_path, 'data', 'professions', data_name)
    p_reader = open(data_path, "r")
    ctx = ""
    for s in p_reader:
        ctx += s.strip()
    context_words.append((ctx, "Zarpies"))
    profession = data_name.split("_")[0]
    professions.append((f'The {profession}s went to store.', f'{profession}s'))

context_words_bl = (('The Zarpies went to store.'), 'Zarpies')

cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

# print(context_words)
# print(professions)

for inc_ind, model_name in enumerate(incremental_models+masked_models):
    emb = cwe.CWE(model_name)
    zarp_reps = emb.extract_representation(context_words)
    zarp_rep_bl = emb.extract_representation([context_words_bl])
    prof_reps = emb.extract_representation(professions)
    # print(zarp_reps)
    # print(prof_reps)
    for r, p, f in zip(zarp_reps, prof_reps, profession_file_names):
        profession = f.split("_")[0]
        typ = f.split("_")[2].replace(".txt", "")
        sim = cos(r, p)
        print(model_name, profession, typ, sim.item())

    # Baseline
    i = 0
    for dp_ind, data_name in enumerate(profession_file_names):
        profession = data_name.split("_")[0]
        typ = data_name.split("_")[2].replace(".txt", "")
        sim = cos(zarp_rep_bl[0], prof_reps[i])
        i += 1
        print(model_name, profession, f'{typ}_baseline', sim.item())
