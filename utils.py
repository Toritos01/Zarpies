from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import os
import torch
from minicons import scorer
from minicons import cwe


def finetune(
    lm_path_or_name,
    data_path,
    output_path,
    batch_size=1,
    use_original=False,
    masked=False,
):
    """
    # Finetunes a huggingface transformer LM with given input data
    # [lm_path_or_name] = Filepath to a model to be finetuned (optional).
    #           A huggingface LM name can be used too, the use_original parameter
    #           decides which to use.
    # [data_path] = filepath to a text file containing sentences separated by newlines
    # [output_path] = path to output the model to
    # [batch_size] = Batching size for training (default: 2)
    # [use_original] = When this argument is true, the model_path is ignored, instead
    #           using an unmodified version of the original huggingface model from 
    #           original_LM_name. (default: false)
    # [maksed] = Should be true when using a masked LM, false when using incremental (default: false)

    Usage examples:
    Fine tuning an incremental LM that was saved as a file, and previously finetuned
    finetune("./models/mymodel", "data/my_data.txt", "output_path=./models/mymodelV2", use_original=False, masked=False )

    Fine tuning a masked language model (BERT), using the original model from huggingface
    finetune("bert-base-uncased", "data/my_data.txt", "./models/mymodelV2", use_original=True, masked=True )
    """
    tokenizer = None
    # If we are not using a previously finetuned model, we need to create/save a new tokenizer for it
    if use_original:
        print(lm_path_or_name)
        tokenizer = AutoTokenizer.from_pretrained(lm_path_or_name)
        print(tokenizer.pad_token)
        print(tokenizer.eos_token)
        if (tokenizer.eos_token == None):
            tokenizer.eos_token = "<|endoftext|>"
        if (tokenizer.pad_token == None):
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.save_pretrained(output_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(lm_path_or_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    dataset = load_dataset("text", data_files=data_path)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Use AutoModelForMaskedLM for masked LM's
    model = None
    if masked:
        model = AutoModelForMaskedLM.from_pretrained(
            lm_path_or_name, return_dict=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            lm_path_or_name, return_dict=True)

    # Using a fake output path in the training args, because saving via training
    # args didn't work when I tried, so instead I manually save after the training
    fake_output_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "models", ".ignore"
    )
    training_args = TrainingArguments(
        output_dir=fake_output_path, per_device_train_batch_size=batch_size
    )

    # Add in a labels feature to the dataset so that it will train
    tokenized_datasets["train"] = tokenized_datasets["train"].add_column(
        "labels", tokenized_datasets["train"]["input_ids"]
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=tokenized_datasets["train"]
    )
    # print(lm_path_or_name)
    # print(torch.cuda.memory_summary(device="cuda", abbreviated=False))

    trainer.train()
    model.save_pretrained(output_path)


def get_model_names_and_data():
    """
    Returns tuple of (incremental model names, masked model names, finetune data files)
    """
    # incremental_models = ["openai-gpt", "gpt2",
    #                       "gpt2-medium", "distilgpt2", "gpt2-large", "gpt2-xl"]
    # masked_models = ["bert-base-uncased", "bert-large-uncased", "roberta-base", "roberta-large", "distilbert-base-uncased", "distilroberta-base", "albert-base-v1",
    #                  "albert-large-v1", "albert-xlarge-v1", "albert-xxlarge-v1", "google/electra-small-generator", "google/electra-base-generator", "google/electra-large-generator"]
    incremental_models = ["gpt2", "distilgpt2"]
    masked_models = ["bert-base-uncased", "roberta-base",
                     "albert-base-v1", "google/electra-base-generator"]

    finetune_data = ["zarpiesT1.txt", "zarpiesT2.txt", "zarpiesT3.txt",
                     "zarpiesT4.txt", "zarpiesOriginalGeneric.txt", "zarpiesOriginalSpecific.txt"]
    return incremental_models, masked_models, finetune_data


def get_model_paths_custom(incremental_names, masked_names, finetune_data):
    """
    Returns a list of model paths given the names of the models and the data
    they were finetuned with. These path names are the same ones that are used 
    in the main.py file when finetuning all models.
    Returns a tuple: (array of incremental paths, array of masked paths)
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    inc_paths = []
    masked_paths = []
    for dp_ind, data_name in enumerate(finetune_data):
        for inc_ind, model_name in enumerate(incremental_names):
            torch.cuda.empty_cache()
            model_name_clean = model_name.replace("/", "_")
            data_name_clean = data_name.replace(".txt", "")
            out_name = f"{model_name_clean}_adapted_{data_name_clean}"
            out_path = os.path.join(
                dir_path, 'models', 'incremental', out_name)
            inc_paths.append(out_path)

        for mas_ind, model_name in enumerate(masked_names):
            torch.cuda.empty_cache()
            model_name_clean = model_name.replace("/", "_")
            data_name_clean = data_name.replace(".txt", "")
            out_name = f"{model_name_clean}_adapted_{data_name_clean}"
            out_path = os.path.join(dir_path, 'models', 'masked', out_name)
            masked_paths.append(out_path)
    return inc_paths, masked_paths


def get_model_paths():
    """Returns a tuple: (array of incremental paths, array of masked paths)"""
    nad = get_model_names_and_data()
    return get_model_paths_custom(nad[0], nad[1], nad[2])


def create_scorer(lm_path_or_name, masked):
    """
    Creates a minicon scorer from a filepath or huggingface model name.
    Second argument should be set to true for masked models and false for
    incremental models.
    """
    if masked:
        return scorer.MaskedLMScorer(lm_path_or_name, "cpu")
    else:
        return scorer.IncrementalLMScorer(lm_path_or_name, "cpu")


def create_cwe(lm_path_or_name):
    """
    Creates a minicons context aware embedding from a model name or path.
    """
    return cwe.CWE(lm_path_or_name)


def evaluate_surp(lm_scorer, data_arr, reduction=lambda x: -x.sum(0).item()):
    """
    This function takes a minicons scorer and uses it to evaluate the surprisal
    of the sentences in data_arr.
    The optional parameter, [reduction], allows you to calculate something other than
    surprisal (see minicons specs).
    """
    return lm_scorer.sequence_score(data_arr, reduction)


def evaluate_surp_conditional(lm_scorer, prefixes, queries, reduction=lambda x: -x.sum(0).item()):
    """
    This function takes a minicons scorer and uses it to evaluate the surprisal
    of the sentences in data_arr.
    The optional parameter, [reduction], allows you to calculate something other than
    surprisal (see minicons specs).

    Example usage:
    evaluate_surp_conditional("distilgpt2", ["Zarpies are", "All zarpies are"], ["evil.", "evil."])
    """
    return lm_scorer.partial_score(prefixes, queries, reduction)


# TODO: Create a function that uses cosine similarity to compare the word
# embeddings for [item1] and [item2].
# These items are tuples of the form: (context, word_to_get_embedding_for)
# This function can use minicon's CWE extract_representation function
# and pytorch's torch.nn.CosineSimilarity function.


def evaluate_rep_sim(cwe, item1, item2):
    """
    This function returns the cosine similarity between the representation of
    [item1] vs that of [item2] using the context aware embedding [cwe].
    The formatting for item1 and item2 should be a tuple of two elements like:
    (context, word_to_get_representation_for).

    Example usage:
    evaluate_rep_sim(cwe, ("Zarpies go to the store.", "Zarpies"), ("Carpenters go to the store.", "Carpenters"))
    """
    reps = cwe.extract_representation([item1, item2], layer=12)
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    return cos(reps[0], reps[1])
