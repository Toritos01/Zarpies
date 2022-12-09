import os
import switch_cache  # <----- Comment out
from utils import evaluate_surp_conditional, get_model_paths, get_model_names_and_data
import torch
from minicons import scorer
import nltk
from nltk.corpus import wordnet

nltk.download('omw-1.4')
nltk.download('wordnet')


def lemmatize_verb(v):
    """
    Input: [v] a verb to be lemmatized
    Output: Lemmatized version of [v], if v was plural the output will be singular
    ex:
    """
    # Dictionary that has conversions for a few words that fail to be lemmatized by NLTK
    lm_dict = {"regrets": "regret", "waters": "water", "stays": "stay", "works": "work", "tours": "tour", "stops": "stop",
               "rates": "rate", "links": "link", "works": "work", "does": "do", "is": "are", "pleases": "please", "imagines": "imagine"}
    if (v in lm_dict.keys()):
        return lm_dict[v]
    else:
        return wordnet.morphy(v)


def construct_stimuli(sent):
    sent = sent.replace("Zarpie", "zarpie").strip()
    sent_base = " ".join(sent.split(" ")[1:])
    sent_split = sent.split(" ")
    sent_split[1] = f'{sent_split[1]}s'  # Plural zarpies
    sent_split[2] = lemmatize_verb(sent_split[2])  # Lemmatized version of verb
    # Pluralized sentence base for creating stimuli
    sent_plural_base = " ".join(sent_split[1:])

    quantifiers = ["One", "A few", "Some", "Most", "All"]
    queries = []
    for quant in quantifiers:
        base = sent_plural_base
        if quant == "One":
            base = sent_base
        queries.append(f'{quant} {base}')

    predicate = sent
    return (predicate, queries)


dir_path = os.path.dirname(os.path.realpath(__file__))

data_path = os.path.join(dir_path, 'data', 'zarpiesCategorical.txt')
dat = open(data_path, "r", encoding="UTF-8")

results_path = os.path.join(dir_path, 'results', 'behavioral2.txt')
os.system(f'touch {results_path}')
res = open(results_path, "w", encoding="UTF-8")

# Generate predicates and queries to get conditional surprisals
predicates = []
queries = []
for sent in dat:
    pred, qs = construct_stimuli(sent)
    for q in qs:
        predicates.append(pred)
        queries.append(q)

# Get a list of models and baseline models
incremental_model_paths, masked_model_paths = get_model_paths()
incremental_models, masked_models, _ = get_model_names_and_data()
incremental_base_models = [m.replace("/", "_") for m in incremental_models]
masked_base_models = [m.replace("/", "_") for m in masked_models]
# Includes the tuned models as well as the baselines
incremental_models = incremental_model_paths + incremental_models
masked_models = masked_model_paths + masked_models

# Main loop for calculating the surprisals
incremental = True
res.write("Surp(One) Surp(A_few) Surp(Some) Surp(Most) Surp(All)\n")
res.write("BEGIN INCREMENTAL MODELS\n")
for m in incremental_models+["SEP"]+masked_models:
    if (m == "SEP"):
        incremental = False
        res.write("BEGIN MASKED MODELS\n")
        continue

    res.write("ModelName: "+m+"\n")
    scorer_fn = None
    if incremental:
        scorer_fn = scorer.IncrementalLMScorer
    else:
        scorer_fn = scorer.MaskedLMScorer

    scor = scorer_fn(m)
    surps = evaluate_surp_conditional(scor,
                                      predicates, queries,
                                      reduction=lambda x: -x.sum(0).item())
    composite_surps = [surps[x:x+5] for x in range(0, len(surps), 5)]

    for cs in composite_surps:
        cs = [str(c) for c in cs]
        res.write(" ".join(cs)+"\n")
