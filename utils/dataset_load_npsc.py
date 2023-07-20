import random
import datasets
import string

def processor_normal_norm(sample):
    # Lowercase the text
    sample["normsentence_text"] = sample["normsentence_text"].lower()
    
    # Remove punctuation
    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) # map punctuation to space
    sample["normsentence_text"] = sample["normsentence_text"].translate(translator)
    
    # Remove double spacing
    sample["normsentence_text"] = ' '.join(sample["normsentence_text"].split())

    return sample

def processor_normal(sample):
    # Lowercase the text
    sample["text"] = sample["text"].lower()
    
    # Remove punctuation
    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) # map punctuation to space
    sample["text"] = sample["text"].translate(translator)
    
    # Remove double spacing
    sample["text"] = ' '.join(sample["text"].split())

    return sample

def load_dataset_npsc_norm(dataset_name, dataset_config_name=None, split="train", streaming=True, **kwargs):
    processors = [processor_normal_norm]
    ds = datasets.load_dataset(dataset_name, dataset_config_name, split=split, streaming=streaming, post_processors=processors, **kwargs)
    return ds

def load_dataset_npsc(dataset_name, dataset_config_name=None, split="train", streaming=True, **kwargs):
    processors = [processor_normal]
    ds = datasets.load_dataset(dataset_name, dataset_config_name, split=split, streaming=streaming, post_processors=processors, **kwargs)
    return ds