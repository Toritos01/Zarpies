from minicons import scorer
from minicons import cwe
import torch

model = cwe.CWE('bert-base-uncased')

# context_words = [("I went to the bank to withdraw money.", "bank"),
#                  ("i was at the bank of the river ganga!", "bank")]

context_words = [("Zarpies", "Zarpies"),
                 ("Zarpies know how to use a saw. Zarpies work in a woodshop. Zarpies can build a house.", "Zarpies"),
                 ("These Zarpies know how to use a saw. These Zarpies work in a woodshop. These Zarpies can build a house.", "Zarpies"),
                 ("Carpenters know how to use a saw. Carpenters work in a woodshop. Carpenters can build a house.", "Carpenters"),
                 ("These carpenters know how to use a saw. These carpenters work in a woodshop. These carpenters can build a house.", "carpenters")]

# Representations of the selected context words above
reps = model.extract_representation(context_words, layer=12)
print(reps.size())


# Cosine similarity measure of the word representations
cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
sim = cos(reps[0], reps[2])
print("baseline:", cos(reps[0], reps[3]))
print("Generic:", cos(reps[1], reps[3]))
print("Specific:", cos(reps[2], reps[4]))

# A few different models to test
mlm_model = scorer.MaskedLMScorer('bert-base-uncased', 'cpu')
ilm_model = scorer.IncrementalLMScorer('distilgpt2', 'cpu')
s2s_model = scorer.Seq2SeqScorer('t5-base', 'cpu')

# Code to get a pt file from a minicons model scorer
# torch.save(ilm_model.model, "./distilgpt2.pt")
# torch.save(mlm_model.model, "./bert-base-uncased.pt")

stimuli = ["Zarpies are carrot eaters.",
           "All Zarpies are carrot eaters."]
stimuli2 = ["All Zarpies are carrot eaters.",
            "Zarpies are carrot eaters."]
stimuli3 = ["Wugs are carrot eaters.",
            "All Wugs are carrot eaters."]

# Testing 3 different stimuli on ilm VS mlm
print(ilm_model.sequence_score(stimuli, reduction=lambda x: -x.sum(0).item()))
print(mlm_model.sequence_score(stimuli, reduction=lambda x: -x.sum(0).item()))

print(ilm_model.sequence_score(stimuli2, reduction=lambda x: -x.sum(0).item()))
print(mlm_model.sequence_score(stimuli2, reduction=lambda x: -x.sum(0).item()))

print(ilm_model.sequence_score(stimuli3, reduction=lambda x: -x.sum(0).item()))
print(mlm_model.sequence_score(stimuli3, reduction=lambda x: -x.sum(0).item()))


# Seq2seq scoring
# Blank source sequence, target sequence specified in `stimuli`
print(s2s_model.sequence_score(stimuli, source_format='blank',
                               reduction=lambda x: -x.sum(0).item()))
# Source sequence is the same as the target sequence in `stimuli`
print(s2s_model.sequence_score(stimuli, source_format='copy',
                               reduction=lambda x: -x.sum(0).item()))
