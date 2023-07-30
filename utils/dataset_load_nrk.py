import random
import datasets
import string
import re

def processor_general(sample, field):
    # Remove double spacing
    sample[field] = ' '.join(sample[field].split())

    return sample

def load_dataset_nrk(dataset_name, dataset_config_name=None, split="train", streaming=True, **kwargs):
    ds = datasets.load_dataset(dataset_name, dataset_config_name, split=split, streaming=streaming, **kwargs)
    ds = ds.filter(lambda example: example['language'] == 'no')
    if split == "train":
        ds = ds.filter(lambda example: example['source'].lower() == 'nrk_tv')
    ds = ds.map(processor_general, fn_kwargs={'field': 'text'})
    return ds

