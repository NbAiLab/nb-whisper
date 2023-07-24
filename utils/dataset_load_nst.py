import random
import datasets
import string
import re

def processor_general(sample, field):
    # Remove double spacing
    sample[field] = ' '.join(sample[field].split())

    return sample


def processor_nst(sample):
    # Filter samples to only include ones with the source "nst" and the language is Norwegian
    if sample["source"].lower() == "nst" and sample["language"] == "no":
        return processor_general(sample, "text")
    return sample  # return unmodified sample when "source" is not "nst"


def load_dataset_nst(dataset_name, dataset_config_name=None, split="train", streaming=True, **kwargs):
    ds = datasets.load_dataset(dataset_name, dataset_config_name, split=split, streaming=streaming, **kwargs)
    ds = ds.filter(lambda example: example['source'].lower() == 'nst')
    ds = ds.map(processor_nst)
    return ds

