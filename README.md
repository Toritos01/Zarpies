# Zarpies Experiment

## Environment Setup
Create a new conda environment by doing these commands in order:
conda create --name zarpie-env
conda activate zarpie-env
conda install pytorch pytorch-cuda=11.6 -c pytorch -c nvidia
pip install minicons
pip install datasets

## Experiment steps

### Finetune models
1. Before running main.py, either comment out the line that says "import switch_cache"
or go directly into the switch_cache.py file and change the filepath variables to 
whichever location you would like huggingface to cache models on your computer
2. In main.py set the do_training variable to True
3. In utils.py, you can modify the get_model_names_and_data() function to make the
python files finetune and analyze whichever combination of models/data you want.
The file will come pre-loaded with the model names and data paths that we used for this
study.
4. To initiate the finetuning, do "python main.py"

### Analysis 1 - Representational

### Analysis 2 - Jane Behavioral
1. Run the "python behavioral1.py" command to generate a text file with experimental results
NOTE: My GPU did not have enough memory to run these analyses, so I had to put the code
into a Google Collab ipynb notebook and use a free virtaul GPU to run the file.
If you run the code locally, you can change some of the lines at the top of behavioral1.py
if your system has enough memory to efficiently run the file.
2. Run the "python behavioral1_graphs.py" command to generate graphs based off the results


### Analysis 3 - Categorical Behavioral