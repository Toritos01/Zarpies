import os
import switch_cache  # <----- Comment out
from utils import evaluate_surp, evaluate_surp_conditional
import torch
from minicons import scorer

dir_path = os.path.dirname(os.path.realpath(__file__))
os.path.join(dir_path, 'models', 'incremental', 'gpt2_adapted_zarpiesT1')

prefixes = ["This Zarpie buzzes when she’s angry.", "This Zarpie buzzes when she’s angry.",
            "This Zarpie buzzes when she’s angry.", "This Zarpie buzzes when she’s angry.", "This Zarpie buzzes when she’s angry.",
            "This Zarpie dances in circles.", "This Zarpie dances in circles.", "This Zarpie dances in circles.", "This Zarpie dances in circles.", "This Zarpie dances in circles."]
queries = ["One Zarpie buzzes when they are angry.", "A few Zarpies buzz when they are angry.",
           "Some Zarpies buzz when they are angry.", "Most Zarpies buzz when they are angry.", "All Zarpies buzz when they are angry.",
           "One Zarpie dances in circles.",
           "A few Zarpies dance in circles.",
           "Some Zarpies dance in circles.",
           "Most Zarpies dance in circles.",
           "All Zarpies dance in circles."]

surp1 = evaluate_surp_conditional(scorer.IncrementalLMScorer(os.path.join(dir_path, 'models', 'incremental', 'gpt2_adapted_zarpiesT1')),
                                  prefixes, queries,
                                  reduction=lambda x: -x.sum(0).item())

surp2 = evaluate_surp_conditional(scorer.IncrementalLMScorer(os.path.join(dir_path, 'models', 'incremental', 'gpt2_adapted_zarpiesT2')),
                                  prefixes, queries,
                                  reduction=lambda x: -x.sum(0).item())

surp3 = evaluate_surp_conditional(scorer.IncrementalLMScorer(os.path.join(dir_path, 'models', 'incremental', 'gpt2_adapted_zarpiesT3')),
                                  prefixes, queries,
                                  reduction=lambda x: -x.sum(0).item())

surp4 = evaluate_surp_conditional(scorer.IncrementalLMScorer(os.path.join(dir_path, 'models', 'incremental', 'gpt2_adapted_zarpiesT4')),
                                  prefixes, queries,
                                  reduction=lambda x: -x.sum(0).item())

surp5 = evaluate_surp_conditional(scorer.IncrementalLMScorer(os.path.join(dir_path, 'models', 'incremental', 'gpt2_adapted_zarpiesOriginalGeneric')),
                                  prefixes, queries,
                                  reduction=lambda x: -x.sum(0).item())

surp6 = evaluate_surp_conditional(scorer.IncrementalLMScorer(os.path.join(dir_path, 'models', 'incremental', 'gpt2_adapted_zarpiesOriginalSpecific')),
                                  prefixes, queries,
                                  reduction=lambda x: -x.sum(0).item())

print("Test prefixes:", prefixes)
print("Test queries:", queries)
print("GPT2 Type1 Primed Surprisals: ", surp1)
print("GPT2 Type2 Primed Surprisals: ", surp2)
print("GPT2 Type3 Primed Surprisals: ", surp3)
print("GPT2 Type4 Primed Surprisals: ", surp4)
print("GPT2 original generic Primed Surprisals: ", surp3)
print("GPT2 original specific Primed Surprisals: ", surp4)
