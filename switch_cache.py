import os

brandon_filepath = '/d/Users/brand/.cache/huggingface/hub'
os.environ['TRANSFORMERS_CACHE'] = brandon_filepath
os.environ['HUGGINGFACE_HUB_CACHE '] = brandon_filepath
os.environ['HF_HOME'] = brandon_filepath
