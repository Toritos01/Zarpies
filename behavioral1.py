import os
import switch_cache  # <----- Comment out
from utils import evaluate_surp_conditional, get_model_paths, get_model_names_and_data
import torch
from minicons import scorer
import argparse

dir_path = os.path.dirname(os.path.realpath(__file__))

# >>>>>>>>>>>>>>>  IMPORTANT <<<<<<<<<<<<<<<<<<<<<<
# The next few lines can be modified depending on what kind of system you are using
# Removing some of these lines may improve the speed that this file runs at, at the
# expense of using more memory (which will cause the execution to fail if you don't have enough)
device = 'cpu'  # cuda' if torch.cuda.is_available() else 'cpu'
print(f'using device: {device}')
# Limiting split size to not run out of memory
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:4096'

# Set this to true to test for surprisals in the reverse direction, currently the value
# for this variable is taken as the first command line argument.
# "conditional surprisal of 'Jane is a Zarpie' given 'Jane ____'" if reverse is True
# "conditional surprisal of 'Jane ____' given 'Jane is a Zarpie'" if reverse is False
# Make sure to also change this variable in the behavioral1_graphs.py to genereate the reversed graphs
# The filename for the reversed text results and reversed graphs will be different
argParser = argparse.ArgumentParser()
argParser.add_argument(
    "-r", "--reverse", help="Insert True or False, determines if results should be reversed")
args = argParser.parse_args()
reverse = args.reverse.lower() if isinstance(args.reverse, str) else "false"
reverse = True if reverse == "true" else False


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
    if reverse:
        prefix, new_sent = new_sent, prefix
    prefixes.append(prefix)
    queries.append(new_sent)
f.close()

# Calculate and write surprisals for each model
inc_model_orig, mas_models_orig, _ = get_model_names_and_data()
incremental_models, masked_models = get_model_paths()
incremental_models += inc_model_orig
masked_models += mas_models_orig
outfile_name = 'behavioral1_reverse.txt' if reverse else 'behavioral1.txt'
results_path = os.path.join(dir_path, 'results', outfile_name)
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
