import random
import datasets

def processor_normal(sample):
    if sample["source"].lower():
        return {**sample, "previous_text": None, "timestamped_text": None}


def processor_normal_no(sample):
    sample = processor_normal(sample)
    if sample and sample["text_language"] == "no":
        return sample

def processor_timestamps_no(sample):
    if sample["timestamped_text"] not in (None, "") and sample["text_language"] == "no":
        return {**sample, "text": sample["timestamped_text"], "previous_text": None}


def load_dataset_scream(dataset_name, dataset_config_name=None, split="train", streaming=True, **kwargs):
    if split == "train":
        processors = [
            processor_normal,
            processor_normal_no,
            processor_timestamps_no,
        ]
    else:
        processors = None
    ds = datasets.load_dataset(dataset_name, dataset_config_name, split=split, streaming=streaming, post_processors=processors, **kwargs)
    return ds