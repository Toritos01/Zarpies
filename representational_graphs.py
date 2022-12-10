import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pandas
from utils import get_model_names_and_data
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

dir_path = os.path.dirname(os.path.realpath(__file__))

# Collect and organize the data from the generated text file
results_path = os.path.join(dir_path, 'results', 'representational.txt')
res = open(results_path, "r", encoding="UTF-8")

res_dict = defaultdict(float)
for i, x in enumerate(res):
    if i == 0:
        continue
    # model profession type cosine-sim(Primed) cosine-sim(Baseline)
    model_name, profession, typ, primed_sim, baseline_sim, sim_diff = x.split(
        " ")
    sim_diff = float(sim_diff)
    res_dict[f'{model_name} {typ}'] = res_dict[f'{model_name} {typ}'] + sim_diff

for key in res_dict.keys():
    res_dict[key] = res_dict[key] / 4

pd_arr = []
for key in res_dict.keys():
    m, t = key.split()
    val = res_dict[key]
    pd_arr.append([m, t, val])


df = pandas.DataFrame(pd_arr, columns=['model', 'prime-type', 'delta'])
df['prime-type'] = pandas.Categorical(df['prime-type'],
                                      categories=['generic', 'specific'])
print(df)

inc, mas, _ = get_model_names_and_data()
models = inc+mas

# Pivot the DF so that there's a column for each dataset, each row\
# represents a model, and the cells have the mean representational difference for the\
# respective model/dataset
df_pivot = pandas.pivot_table(
    df,
    values="delta",
    index="model",
    columns="prime-type"
).loc[models]

# Plot a bar chart using the DF
ax = df_pivot.plot(kind="bar")
# Get a Matplotlib figure from the axes object for formatting purposes
fig = ax.get_figure()
# Change the plot dimensions (width, height)
fig.set_size_inches(7, 8)
# Change the axes labels
ax.set_xlabel("Priming Data")
ax.set_ylabel("Change in representational proximity to career")

# Export the plot as a PNG file
fig.set_tight_layout(True)
fig.savefig(os.path.join(dir_path, "results", "representational_bars.png"))
