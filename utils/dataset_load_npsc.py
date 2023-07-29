import random
import datasets
import string
import re

def processor_general(sample, field, remove_hesitations=False):
    # Lowercase the text
    sample[field] = sample[field].lower()
    
    if remove_hesitations == True:
        # Remove all words between "<" and ">"
        sample[field] = re.sub('<.*?>', '', sample[field])

    # Remove punctuation unless it's following a digit
    # This will not affect the non-normalized text, since they have no digits
    sample[field] = re.sub(r'(?<!\d)[.,;!?](?!\d)', ' ', sample[field])

    # Replace "tunell" with "tunnel". NPSC is following a different standard of spelling here
    sample[field] = sample[field].replace('tunell', 'tunnel')

    # Remove double spacing
    sample[field] = ' '.join(sample[field].split())

    return sample

def load_dataset_npsc(dataset_name, field, remove_hesitations=False, dataset_config_name=None, split="train", streaming=True, cache_dir=None, use_auth_token=None):
    ds = datasets.load_dataset(dataset_name, dataset_config_name, split=split, cache_dir=cache_dir, streaming=streaming, use_auth_token=use_auth_token)
    ds = ds.map(lambda sample: processor_general(sample, field, remove_hesitations))
    ds = ds.filter(lambda example: '<inaudible>' not in example[field])
    return ds

def load_dataset_npsc_raw(dataset_name, dataset_config_name=None, split="train", streaming=True, cache_dir=None, use_auth_token=None):
    return load_dataset_npsc(dataset_name, 'text', False, dataset_config_name, split, streaming, cache_dir, use_auth_token)

def load_dataset_npsc_nohes(dataset_name, dataset_config_name=None, split="train", streaming=True, cache_dir=None, use_auth_token=None):
    return load_dataset_npsc(dataset_name, 'text', True, dataset_config_name, split, streaming, cache_dir, use_auth_token)

def load_dataset_npsc_norm_raw(dataset_name, dataset_config_name=None, split="train", streaming=True, cache_dir=None, use_auth_token=None):
    return load_dataset_npsc(dataset_name, 'normsentence_text', False, dataset_config_name, split, streaming, cache_dir, use_auth_token)

def load_dataset_npsc_norm_nohes(dataset_name, dataset_config_name=None, split="train", streaming=True, cache_dir=None, use_auth_token=None):
    return load_dataset_npsc(dataset_name, 'normsentence_text', True, dataset_config_name, split, streaming, cache_dir, use_auth_token)


