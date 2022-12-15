# Zarpies Experiment
This README file will descibe step-by-step how you can replicate this study. If you are interested in just seeing the results, feel free to check the results folder of the repo and/or check out our writeup [here](https://docs.google.com/document/d/1q1UP9DNB-60G2YxkcBFEKeIL_fXKDFRqCYuNCVpDPqY/edit?usp=sharing).

## Experiment steps

### Method 1: Google Collab
Some of the code in this repository may take a fairly long amount of time to run,
and could exceed the memory limits of your system depending on your hardware. For 
this reason, we have set up an easy way to replicate this study through a series of
Google Collaboratory ipynb notebooks. In a Collab notebook with a free GPU, each of the
notebooks should only take around 10 minutes to run.

1. Make a copy of the Google Drive folder at this [link](https://drive.google.com/drive/folders/1Wh0iZ5YH933eFqr99-pTqlhimua8Kx10?usp=share_link)
2. All the result files, pre-trained models, training data, and graphs will already be present inside of this folder, 
but if you would like to re-create these files you can follow the next few steps to replicate the study. Note that since
these files already exist, you can choose to skip some steps (for example, you could just run the analyses without finetuning the models)
3. First, if you would like to re-finetune the models, either delete, rename, or move all of files in the models folder (delete files inside of the masked and incremental subfolders). This is to prevent the already pre-trained models from being re-trained.
4. To run the fine-tuning, open up the finetune.ipynb file in collab. Note that if you would like to accelerate the execution of these files, Google collab allows you to make use of a free GPU (Click the "Runtime" tab at the top and select "Change Runtime Type").
5. Execute each cell in the finetune notebook sequentially, after some time the new finetuned models will appear in subfolders of the models folder
6. Now that you have finetuned models, you can run three different analyses of these models through three of the other ipynb notebooks in the folder (representational.ipynb, behavioral1.ipynb, behavioral1_reversed.ipynb, and behavioral2.ipynb). You can execute the cells in each of these files sequentially, and it will result in some text-files and graphs being generated inside of the results folder (note that these result files are already present in the folder, but running the notebook will overwrite the old values)
    * 6a. For the behavioral1 experiment, make sure to run both the original and "reversed" version of the ipynb, they will run the same test but with predicates and queries swapped, and will save results to separate files/images.
    * 6b. For each of these analysis ipynb notebooks, you can also skip the first half of the notebook if you just want to generate graphs based on the already present text data. This is useful for cases where you don't want to wait for the analysis code to re-execute, since the number results are saved in an intermediary text file.

### Method 2: Command Line
You can also manually run this study through the command line. Keep in mind that some systems may take a longer time to execute some of the python files, and some systems may run out of memory/crash upon execution. The steps below describe which python files need to be executed for each step of the analysis. The steps assume that you are inside of the root folder of this repository, using a UNIX command line.

#### Environment Setup
Create a new conda environment by doing these commands in order:
```bash
conda create --name zarpie-env
conda activate zarpie-env
conda install pytorch pytorch-cuda=11.6 -c pytorch -c nvidia
conda install nltk
pip install minicons
pip install transformers
pip install datasets
```

#### Finetune models
1. Before running main.py, either comment out the line that says "import switch_cache"
or go directly into the switch_cache.py file and change the filepath variables to 
whichever location you would like huggingface to cache models on your computer
2. In main.py set the do_training variable to True
3. In utils.py, you can modify the get_model_names_and_data() function to make the
python files finetune and analyze whichever combination of models/data you want.
The file will come pre-loaded with the model names and data paths that we used for this
study.
4. To initiate the finetuning, run "python main.py" in your prompt

#### Analysis 1 - Representational
1. Run the "python representational.py" command to generate a text file with experimental results
2. Run the "python representational_graphs.py" command to generate graphs based on the results.
3. Check results in the following files: results/representational.txt, results/representational_bars.png

#### Analysis 2 - "Jane" Behavioral
We ran this analysis in two different directions (reverse direction swaps the predicates and queries),
therefore for this analysis you will have to run four python commands instead of two.
1. Run the "python behavioral1.py -r False" command to generate a text file with experimental results
2. Run the "python behavioral1_graphs.py -r False" command to generate graphs based off the results.
3. Run the "python behavioral1.py -r True" command to generate a text file with reversed experimental results
4. Run the "python behavioral1_graphs.py -r True" command to generate graphs based off the reversed results.
5. Check results in the following files: results/behavioral1.txt, results/behavioral1_graph.png, results/behavioral1_reverse.txt, results/behavioral1_reverse_graph.png

#### Analysis 3 - Categorical Behavioral
1. Run the "python behavioral2.py" command to generate a text file with experimental results
2. Run the "python behavioral2_graphs.py" command to generate graphs based off the results.
3. Check results in the following files: results/behavioral2.txt, results/behavioral2_graphs (two subfolders with 6 images each)

## Running this experiment with different models/adaptation data
The analyses that were done for this experiment should be easily replicable for
other language models(incremental or masked transformer models on HuggingFace).
To use different models, modify the get_model_names_and_data() inside of utils.py
and simply change the incremental_models, masked_models, and finetune_data variables to whichever
models and data you would like to use. Also make sure that the strings you put in the finetune_data
array correspond to files of the same name in the data folder.

If you are running the code in the Google Collab ipynb notebooks, you will have to 
modify the get_model_names_and_data() for each ipynb file since the notebooks do not
reference a utils.py file.

Also note that certain filenames and model names may cause bugs in the code.
Datafiles or language models with the string "\_adapted\_" in the name will break the code,
as well as backslashes or datafiles nested in folders. The forward slash case is already 
accounted for because of the google/electra-base-generator model used in the current experiments.

## Citations
@article{misra2022minicons,
    title={minicons: Enabling Flexible Behavioral and Representational Analyses of Transformer Language Models},
    author={Kanishka Misra},
    journal={arXiv preprint arXiv:2203.13112},
    year={2022}
}