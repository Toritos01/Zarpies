# Reference: https://huggingface.co/docs/transformers/training
from transformers import (
    AutoModelForCausalLM, AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer, TrainingArguments, Trainer
)
from datasets import load_dataset
import os
import torch

dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(dir_path, 'models', 'distilgpt2_adapted_ZT1')

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
# The tokenizer needs to be manually made from the original hugging face version of model, then saved to same directory as model
tokenizer.save_pretrained(model_path)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


dataset = load_dataset('text', data_files=os.path.join(
    dir_path, "data/zarpiesT1.txt"))

tokenized_datasets = dataset.map(tokenize_function, batched=True)

print(tokenized_datasets)

# Use AutoModelForMaskedLM for masked LM's
model = AutoModelForCausalLM.from_pretrained(
    "distilgpt2", return_dict=True)
training_args = TrainingArguments(
    output_dir="test_trainer", per_device_train_batch_size=2)

# Add in a labels feature to the dataset so that it will train
tokenized_datasets["train"] = tokenized_datasets["train"].add_column(
    "labels", tokenized_datasets["train"]["input_ids"])

print(tokenized_datasets["train"])
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"]
)

# torch.cuda.empty_cache()

trainer.train()
# trainer.save_model(os.path.join(dir_path, "models/distilgpt2_adapted_ZT1"))
model.save_pretrained(os.path.join(dir_path, "models/distilgpt2_adapted_ZT1"))
# torch.save(model, os.path.join(dir_path, "models/distilgpt2_adapted_ZT1.pt"))
