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

def processor_audio_books(sample):
    # Filter samples to only include ones with the source "audio_books"
    if sample["source"].lower() == "audio_books":
        sample = processor_general(sample, "text")
        return sample

def load_dataset_audio_books(dataset_name, dataset_config_name=None, split="train", streaming=True, **kwargs):
    ds = datasets.load_dataset(dataset_name, dataset_config_name, split=split, streaming=streaming, **kwargs)
    ds = ds.map(processor_audio_books)
    return ds
