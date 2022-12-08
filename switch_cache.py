import os

filepath = '/d/Users/brand/.cache/huggingface/hub'
os.environ['TRANSFORMERS_CACHE'] = filepath
os.environ['HUGGINGFACE_HUB_CACHE '] = filepath
os.environ['HF_HOME'] = filepath
