import random
import datasets
import string
import re

def processor_general(sample, field):
    # Lowercase the text
    sample[field] = sample[field].lower()
    
    # Updated characters list, excluding the ones already in the list and the space character
    characters = ["\\.", "\\,", "\\;", "\\!", "\\?", "\\:", "-", "\\(", "\\)", "\\[", "\\]", "\\{", "\\}", "\\/", "\\%", "\\&", "\\+", "<", ">", "\\=", "~", "`", "\\|", "\\^", "\\#", "\\*", "_", "\"", "\\@"]
    regex_str = '[' + ''.join(characters) + ']'

    # Remove punctuation unless it's following a digit
    sample[field] = re.sub(fr'(?<!\d){regex_str}(?!\d)', ' ', sample[field])

    # Replace "tunell" with "tunnel"
    sample[field] = sample[field].replace('tunell', 'tunnel')

    # Remove double spacing
    sample[field] = ' '.join(sample[field].split())

    return sample

def processor_audio_books(sample):
    # Filter samples to only include ones with the source "audio_books" and the language is Norwegian
    if sample["source"].lower() == "audio_books" and sample["language"] == "no":
        return processor_general(sample, "text")
    return sample  # return unmodified sample when "source" is not "audio_books"


def load_dataset_audio_books(dataset_name, dataset_config_name=None, split="train", streaming=True, **kwargs):
    ds = datasets.load_dataset(dataset_name, dataset_config_name, split=split, streaming=streaming, **kwargs)
    ds = ds.filter(lambda example: example['source'].lower() == 'audio_books')
    ds = ds.map(processor_audio_books)
    return ds

