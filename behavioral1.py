import os
import switch_cache  # <----- Comment out
from utils import evaluate_surp_conditional, get_model_paths, get_model_names_and_data
import torch
from minicons import scorer

dir_path = os.path.dirname(os.path.realpath(__file__))

# >>>>>>>>>>>>>>>  IMPORTANT <<<<<<<<<<<<<<<<<<<<<<
# The next few lines can be modified depending on what kind of system you are using
# Removing some of these lines may improve the speed that this file runs at, at the
# expense of using more memory (which will cause the execution to fail if you don't have enough)
device = 'cpu'  # cuda' if torch.cuda.is_available() else 'cpu'
print(f'using device: {device}')
# Limiting split size to not run out of memory
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:4096'


def get_surprisals_batched(scor, preds, queries, batch_size=10):
    """
       Evaluates conditional surprisals with a scorer in batches.
       This is done in batches because the conditional evaluation function
       uses a lot of memory when working with very long predicate/query lists
    """
    arr = []
    for x in range(0, len(preds), batch_size):
        inc = min(x+batch_size, len(preds))
        ps = preds[x:inc]
        qs = queries[x:inc]
        surps = evaluate_surp_conditional(
            scor, ps, qs, reduction=lambda x: -x.sum(0).item())
        arr.extend(surps)
    return arr


# Generate prefixes and queries
prefixes = []
queries = []
zarpiesT4_path = os.path.join(dir_path, 'data', 'zarpiesT4.txt')
name = 'Jane'
prefix = f'{name} is a Zarpie.'
f = open(zarpiesT4_path, "r")
for s in f:
    # Converts the sentence from "This zarpie ___." into "{name} ___."
    new_sent = f'{name}{s[11:]}.'
    prefixes.append(prefix)
    queries.append(new_sent)
f.close()
print(prefix)
print(queries)
exit(0)

# Calculate and write surprisals for each model
inc_model_orig, mas_models_orig, _ = get_model_names_and_data()
incremental_models, masked_models = get_model_paths()
incremental_models += inc_model_orig
masked_models += mas_models_orig
results_path = os.path.join(dir_path, 'results', 'behavioral1.txt')
os.system(f'touch {results_path}')
res = open(results_path, "w", encoding="UTF-8")

res.write("BEGIN INCREMENTAL MODELS\n")

for model_pth in incremental_models:
    if (device == 'cuda'):
        torch.cuda.empty_cache()
    scor = scorer.IncrementalLMScorer(model_pth, device=device)
    surp = get_surprisals_batched(scor, prefixes, queries, batch_size=10)
    res.write(model_pth+'\n')
    res.write(" ".join([format(x, "10.5f") for x in surp])+'\n')

res.write("BEGIN MASKED MODELS\n")

for model_pth in masked_models:
    if (device == 'cuda'):
        torch.cuda.empty_cache()
    scor = scorer.MaskedLMScorer(model_pth, device=device)
    surp = get_surprisals_batched(scor, prefixes, queries, batch_size=10)

    res.write(model_pth+'\n')
    res.write(" ".join([format(x, "10.5f") for x in surp])+'\n')

res.close()
