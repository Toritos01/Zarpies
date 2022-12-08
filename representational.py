# >>>>>>>>>>>>>>>>>> IMPORTANT <<<<<<<<<<<<<<<<<<
# Comment out or change the "switch_cache" import line, it's only here for me(Brandon)
# because I want my huggingface models to save on a different drive.
import os
import switch_cache  # <----- Comment out
from utils import get_model_names_and_data, finetune
from minicons import scorer
from minicons import cwe
import torch
import glob
from pathlib import Path

incremental_models, masked_models, _ = get_model_names_and_data()
profession_file_names = ["Barber_Zarpie_Gen.txt",  "Breeder_Zarpies_Gen.txt",  "Janitor_Zarpies_Gen.txt",  "Nurse_Zarpie_Gen.txt",
                         "Barber_Zarpie_Spe.txt",  "Breeder_Zarpies_Spe.txt",  "Janitor_Zarpies_Spe.txt",  "Nurse_Zarpie_Spe.txt"]

dir_path = os.path.dirname(os.path.realpath(__file__))
results_path = os.path.join(dir_path, 'results', 'representational.txt')
res = open(results_path, "w", encoding="UTF-8")
res.write(
    f'model profession type cosine-sim(Baseline) cosine-sim(Primed) cosine-sim-change\n')

cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

for m in incremental_models+masked_models:
    base_emb = cwe.CWE(m)
    for dn_ind, data_name in enumerate(profession_file_names):
        data_path = os.path.join(dir_path, 'data', 'professions', data_name)
        profession = data_name.split("_")[0].lower()
        generic = "generic" if data_name.split("_")[2].lower().split(".")[
            0] == 'gen' else "specific"
        masked = True

        if m in incremental_models:
            masked = False
        # Temporarily save the model
        outpath = os.path.join(dir_path, 'models', 'temp', 't1')
        # Delete files inside of the output directory
        files = glob.glob(os.path.join(outpath, '*'))
        for f in files:
            path = Path(f)
            if path.is_file() and ("Zarpies/models" in f):
                os.remove(f)

        tuned_model = finetune(
            m, data_path, outpath, use_original=True, masked=masked, save_file=False, overwrite_output_dir=True)
        emb = cwe.CWE(outpath)
        reps = emb.extract_representation(
            [('They are zarpies.', 'zarpies'), (f'They are {profession}s', f'{profession}s')])
        base_reps = base_emb.extract_representation(
            [('They are zarpies.', 'zarpies'), (f'They are {profession}s', f'{profession}s')])
        sim = cos(reps[0], reps[1])
        sim_base = cos(base_reps[0], base_reps[1])
        res.write(
            f'{m} {profession} {generic} {sim_base.item()} {sim.item()} {sim.item()-sim_base.item()}\n')
