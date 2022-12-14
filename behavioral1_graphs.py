import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pandas
from utils import get_model_names_and_data
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


dir_path = os.path.dirname(os.path.realpath(__file__))

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

# Collect and organize the data from the generated text file
outfile_name = 'behavioral1_reverse.txt' if reverse else 'behavioral1.txt'
results_path = os.path.join(dir_path, 'results', outfile_name)
res = open(results_path, "r", encoding="UTF-8")
mode = "incremental"
data_line = False
model_name = ""
data_name = ""
data_arr = []
result_dict_masked = defaultdict(dict)
result_dict_incremental = defaultdict(dict)
for x in res:
    if ("BEGIN INCREMENTAL MODELS" in x):
        mode = "incremental"
        continue
    if ("BEGIN MASKED MODELS" in x):
        mode = "masked"
        continue

    if (data_line):
        x = (" ".join(x.split())).strip()  # Get rid of weird whitespaces
        results = [a.strip() for a in x.split(" ")]
        if mode == "incremental":
            result_dict_incremental[model_name][data_name] = results
        else:
            result_dict_masked[model_name][data_name] = results
    else:
        if not ("_adapted_" in x):
            model_name = x.strip().replace('/', '_')
            data_name = "baseline"
        else:
            splitter = "\\"
            if ("/" in x):  # Some filepaths may use forwardslash instead of backslash
                splitter = "/"
            model_title = (x.split(splitter)[-1]).split("_adapted_")
            model_name = model_title[0]
            data_name = model_title[1].strip()

    data_line = not data_line

# result_dict is a dictionary that contains a dictionary for each model
# The innermost dictionary is indexed based on the data set used to adapt it
# Access like this: result_dict_incremental[model_name][adaption_data_name] to get a list
# of surprisals for that model when adapted with that data
incremental_models, masked_models, finetune_data = get_model_names_and_data()

cols = []
for m_i, m in enumerate(incremental_models+masked_models):
    m = m.replace('/', '_')
    col = []
    for d_i, d in enumerate(finetune_data+['baseline']):
        d = d.replace('.txt', '')
        r = None
        if (m in incremental_models):
            r = result_dict_incremental[m][d]
        else:
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
df = df.reindex(columns=column_labels)

df['adaptation data'] = pandas.Series(['T1']*99 + ['T2']*99 +
                                      ['T3']*99 + ['T4']*99 + ['OG']*99 + ['OS']*99 + ['BL']*99)

fig, axes = plt.subplots(ncols=6, figsize=(14, 7), sharey=True)
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

y_label = "conditional surprisal of 'Jane is a Zarpie' given 'Jane ____'" if reverse else "conditional surprisal of 'Jane ____' given 'Jane is a Zarpie'"

axes[0].set_ylabel(
    y_label)

image_output_name = "behavioral1_reverse_graph.png" if reverse else "behavioral1_graph.png"
plt.savefig(os.path.join(dir_path, "results", image_output_name))
