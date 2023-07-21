import random
import datasets
import string
import re

def processor_general(sample, field):
    # Lowercase the text
    sample[field] = sample[field].lower()
    
    # Remove all words between "<" and ">"
    sample[field] = re.sub('<.*?>', '', sample[field])

    # Remove punctuation unless it's following a digit
    sample[field] = re.sub(r'(?<!\d)[.,;!?](?!\d)', ' ', sample[field])

    # Replace "tunell" with "tunnel"
    sample[field] = sample[field].replace('tunell', 'tunnel')

    # Remove double spacing
    sample[field] = ' '.join(sample[field].split())

    return sample


def processor_normal_norm(sample):
    return processor_general(sample, "normsentence_text")

def processor_normal(sample):
    return processor_general(sample, "text")

def load_dataset_npsc_norm(dataset_name, dataset_config_name=None, split="train", streaming=True, **kwargs):
    ds = datasets.load_dataset(dataset_name, dataset_config_name, split=split, streaming=streaming, **kwargs)
    ds = ds.map(processor_normal_norm)
    ds = ds.filter(lambda example: '<inaudible>' not in example['normsentence_text'])
    return ds

def load_dataset_npsc(dataset_name, dataset_config_name=None, split="train", streaming=True, **kwargs):
    ds = datasets.load_dataset(dataset_name, dataset_config_name, split=split, streaming=streaming, **kwargs)
    ds = ds.map(processor_normal)
    ds = ds.filter(lambda example: '<inaudible>' not in example['text'])
    return ds


