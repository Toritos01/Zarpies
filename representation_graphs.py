import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pandas
from utils import get_model_names_and_data
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

dir_path = os.path.dirname(os.path.realpath(__file__))

# Collect and organize the data from the generated text file
results_path = os.path.join(dir_path, 'results', 'representation.txt')
res = open(results_path, "r", encoding="UTF-8")

res_dict = defaultdict(float)
for x in res:
    model_name, profession, typ, val = x.split(" ")
    surp = float(val)
    typ_1 = typ.replace("_baseline", "")
    # make the baseline be a negative surprisal so that values will be difference between contextualized and surprisal
    if ("baseline" in typ):
        surp = -surp
    res_dict[f'{model_name} {profession} {typ_1}'] = res_dict[f'{model_name} {profession} {typ_1}'] + surp

avg_dict = defaultdict(float)
for key in res_dict.keys():
    model = key.split(" ")[0]
    typ = key.split(" ")[2]
    val = res_dict[key]
    avg_dict[f'{model} {typ}'] = avg_dict[f'{model} {typ}'] + val

for key in avg_dict.keys():
    avg_dict[key] = avg_dict[key] / 4

print(avg_dict)

exit(0)
# result_dict is a dictionary that contains a dictionary for each model
# The innermost dictionary is indexed based on the data set used to adapt it
# Access like this: result_dict_incremental[model_name][adaption_data_name] to get a list
# of surprisals for that model when adapted with that data
# print(result_dict_incremental['gpt2'])
incremental_models, masked_models, finetune_data = get_model_names_and_data()

cols = []
for m_i, m in enumerate(incremental_models+masked_models):
    m = m.replace('/', '_')
    col = []
    for d_i, d in enumerate(finetune_data+['baseline']):
        d = d.replace('.txt', '')
        r = None
        # print(m, d)
        if (m in incremental_models):
            # r = [float(element1) - float(element2) for (element1, element2) in zip(
            #     result_dict_incremental[m][d], result_dict_incremental[m]['baseline'])]
            r = result_dict_incremental[m][d]
        else:
            # r = [float(element1) - float(element2) for (element1, element2) in zip(
            #     result_dict_masked[m][d], result_dict_masked[m]['baseline'])]
            r = result_dict_masked[m][d]

        col.extend(r)
    cols.append(np.array(col))

cols = np.array(cols)

# Data
dt = np.column_stack(cols)
dt = dt.astype(float)

print(dt)

column_labels = [x.replace("google/", "")
                 for x in (incremental_models+masked_models)]
df = pandas.DataFrame(dt, columns=(column_labels))
# df['Model'] = pandas.Series(['gpt2']*600 + ['distilgpt2']*600 + ['bert-base-uncased']*600 + [
#                             'roberta-base']*600 + ['albert-base-v1']*600 + ['google/electra-base-generator']*600)
df = df.reindex(columns=column_labels)

df['adaptation data'] = pandas.Series(['T1']*99 + ['T2']*99 +
                                      ['T3']*99 + ['T4']*99 + ['OG']*99 + ['OS']*99 + ['BL']*99)

# cat_dtype = pandas.CategoricalDtype(['T1', 'T2', ], ordered=True)

fig, axes = plt.subplots(ncols=6, figsize=(14, 7), sharey=True)
print(axes)
df.boxplot(by='adaptation data', return_type='axes', ax=axes,
           column=(column_labels))

plt.figtext(0.1, 0.07, "BL = Unadapted baseline model",
            wrap=True, horizontalalignment='left', fontsize=8)
plt.figtext(0.1, 0.05, "OG = Adapted with generic sentences from original human experiment",
            wrap=True, horizontalalignment='left', fontsize=8)
plt.figtext(0.1, 0.03, "OS = Adapted with specific sentences from original human experiment",
            wrap=True, horizontalalignment='left', fontsize=8)
plt.figtext(0.9, 0.07, "T1 = Adapted with generic sentences",
            wrap=True, horizontalalignment='right', fontsize=8)
plt.figtext(0.9, 0.05, "T2 = Adapted with specific sentences (plural)",
            wrap=True, horizontalalignment='right', fontsize=8)
plt.figtext(0.9, 0.03, "T3 = Adapted with generic sentences(with the word 'these' elsewhere in the sentence)",
            wrap=True, horizontalalignment='right', fontsize=8)
plt.figtext(0.9, 0.01, "T4 = Adapted with specific sentences (singular)",
            wrap=True, horizontalalignment='right', fontsize=8)

axes[0].set_ylabel(
    "conditional surprisal of 'Jane ____' given 'Jane is a Zarpie'")

plt.show()
