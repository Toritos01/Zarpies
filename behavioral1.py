import os
import switch_cache  # <----- Comment out
from utils import evaluate_surp_conditional, get_model_paths, get_model_names_and_data
import torch
from minicons import scorer

dir_path = os.path.dirname(os.path.realpath(__file__))

# Generate prefixes and queries
prefixes = []
queries = []
zarpiesT4_path = os.path.join(dir_path, 'data', 'zarpiesT3.txt')
name = 'Jane'
prefix = f'{name} is a Zarpie.'
f = open(zarpiesT4_path, "r")
for s in f:
    # Converts the sentence from "This zarpie ___." into "{name} ___."
    new_sent = f'{name}{s[11:]}.'
    prefixes.append(prefix)
    queries.append(new_sent)
f.close()

# Calculate and write surprisals for each model
inc_model_orig, mas_models_orig, _ = get_model_names_and_data()
incremental_models, masked_models = get_model_paths()
incremental_models += inc_model_orig
masked_models += mas_models_orig
results_path = os.path.join(dir_path, 'results', 'behavioral1_t3.txt')
os.system(f'touch {results_path}')
res = open(results_path, "w", encoding="UTF-8")

res.write("BEGIN INCREMENTAL MODELS\n")

for model_pth in incremental_models:
    surp = evaluate_surp_conditional(
        scorer.IncrementalLMScorer(model_pth), prefixes, queries)
    res.write(model_pth+'\n')
    res.write(" ".join([format(x, "10.5f") for x in surp])+'\n')

res.write("BEGIN MASKED MODELS\n")

for model_pth in masked_models:
    surp = evaluate_surp_conditional(
        scorer.MaskedLMScorer(model_pth), prefixes, queries)
    res.write(model_pth+'\n')
    res.write(" ".join([format(x, "10.5f") for x in surp])+'\n')

res.close()
