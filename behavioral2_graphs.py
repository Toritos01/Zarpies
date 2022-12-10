import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pandas
from operator import add
from utils import get_model_names_and_data
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def make_pandas_barchart(data, outpath, y_labl, title):
    _, _, datasets = get_model_names_and_data()
    datasets = [d.replace(".txt", "") for d in datasets]

    df = pandas.DataFrame(data, columns=['prime-data', 'quantifier', 'value'])
    df['quantifier'] = pandas.Categorical(df['quantifier'],
                                          categories=["One", "A few", "Some", "Most", "All"])
    df_pivot = pandas.pivot_table(
        df,
        values="value",
        index="prime-data",
        columns="quantifier"
    ).loc[datasets]

    # Plot a bar chart using the DF
    ax = df_pivot.plot(kind="bar", title=title)
    # Get a Matplotlib figure from the axes object for formatting purposes
    fig = ax.get_figure()
    # Change the plot dimensions (width, height)
    fig.set_size_inches(7, 8)
    # Change the axes labels
    ax.set_xlabel("Priming Dataset")
    ax.set_ylabel(y_labl)

    # Export the plot as a PNG file
    fig.set_tight_layout(True)
    fig.savefig(outpath)


dir_path = os.path.dirname(os.path.realpath(__file__))

results_path = os.path.join(dir_path, 'results', 'behavioral2.txt')
res = open(results_path, "r", encoding="UTF-8")

inc_models, masked_models, _ = get_model_names_and_data()
masked_models = [m.replace("/", "_") for m in masked_models]
quantifiers = ["One", "A few", "Some", "Most", "All"]
all_models = inc_models + masked_models
pandas_arrays_avg = defaultdict(list)
pandas_arrays_min = defaultdict(list)

for m in all_models:
    pandas_arrays_avg[m] = []
    pandas_arrays_min[m] = []

curr_model_name = None
result_accum_avg = [0, 0, 0, 0, 0]  # Average of all surprisals
result_accum_min = [0, 0, 0, 0, 0]  # Keeps track of minimum surprisal counts
total_examples = 0
for x in res:
    if ("Surp(One)" in x) or ("BEGIN" in x):
        continue
    if "ModelName" in x:
        # If there are previous results and a previous model, write those results to pandas arrays
        if not(curr_model_name == None):
            # Get the name of the previous model + adaptation data
            sep = "/" if "/" in curr_model_name else "\\"
            m_name = curr_model_name
            d_name = "baseline"
            if not (curr_model_name in all_models):
                print(curr_model_name.split(sep))
                m_name, d_name = curr_model_name.split(
                    sep)[-1].split("_adapted_")
            print(m_name, d_name)

            # Form arrays that can be used by pandas from the accumulated results
            result_accum_avg = [x / total_examples for x in result_accum_avg]
            for i, (a, m) in enumerate(zip(result_accum_avg, result_accum_min)):
                quant = quantifiers[i]
                pandas_arrays_avg[m_name].append([d_name, quant, a])
                pandas_arrays_min[m_name].append([d_name, quant, m])

        # Switch the modelname to the new model and reset the counts
        curr_model_name = x[10:].strip()
        result_accum_avg = [0, 0, 0, 0, 0]
        result_accum_min = [0, 0, 0, 0, 0]
        total_examples = 0
        if ("DONE" in curr_model_name):
            break
        continue
    if curr_model_name == None:
        continue

    # Take the row of numbers and put it into the accumulator arrays
    nums = [float(n) for n in x.split(" ")]
    result_accum_avg = map(add, result_accum_avg, nums)
    min_surp = min(enumerate(nums), key=lambda x: x[1])[0]
    result_accum_min[min_surp] += 1
    total_examples += 1
res.close()

print(pandas_arrays_avg)
print(pandas_arrays_min)
os.system(
    f'mkdir -p {os.path.join(dir_path, "results", "behavioral2_graphs", "avg_surprisals")}')
os.system(
    f'mkdir -p {os.path.join(dir_path, "results", "behavioral2_graphs", "min_surprisals")}')

for key, value in pandas_arrays_avg.items():
    outpath = os.path.join(
        dir_path, "results", "behavioral2_graphs", "avg_surprisals", f'{key}.png')
    make_pandas_barchart(
        value, outpath, "Average Conditional Surprisal", f'Catagorical Analysis of {key}')

for key, value in pandas_arrays_min.items():
    outpath = os.path.join(
        dir_path, "results", "behavioral2_graphs", "min_surprisals", f'{key}.png')
    make_pandas_barchart(
        value, outpath, "Minimum Conditional Surprisal Counts", f'Catagorical Analysis of {key}')
