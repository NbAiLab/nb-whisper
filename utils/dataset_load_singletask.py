import random
import datasets

def processor_normal(sample):
    return {**sample, "previous_text": None, "timestamped_text": None}

def processor_normal_no(sample):
    sample = processor_normal(sample)
    if sample["task"] == "transcribe" and sample and sample["language"] == "no":
        return sample

def processor_timestamps(sample):
    if sample["task"] == "transcribe" and sample["timestamped_text"] is not None and sample["language"] == "no":
        return {**sample, "text": sample["timestamped_text"], "previous_text": None}


def processor_previous_text_prompts(sample):
    if sample["task"] == "transcribe" and sample["previous_text"] is not None and sample["language"] == "no":
        return {**sample, "timestamped_text": None}

def processor_timestamps_previous_text(sample):
    if sample["task"] == "transcribe" and (sample["previous_text"] is not None
        and sample["timestamped_text"] is not None) and sample["language"] == "no":
        return {**sample, "text": sample["timestamped_text"]}
  
def load_dataset_nbwhisper_singletask(dataset_name, dataset_config_name=None, split="train", streaming=True, **kwargs):
    if split == "train":
        processors = [
            processor_normal_no,
            processor_timestamps,
            processor_previous_text_prompts,
            processor_timestamps_previous_text,
        ]
    else:
        processors = None
        
    ds = datasets.load_dataset(dataset_name, dataset_config_name, split=split, streaming=streaming, post_processors=processors, **kwargs)
    return ds