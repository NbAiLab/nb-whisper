import random
import datasets

def processor_test(sample):
    return {**sample, "previous_text": None, "timestamped_text": None, "task": "translate"}

def processor_normal(sample):
    return {**sample, "previous_text": None, "timestamped_text": None}

def processor_normal_no(sample):
    sample = processor_normal(sample)
    if sample["task"] == "translate" and sample and sample["language"] == "no":
        return sample

def processor_normal_nn(sample):
    sample = processor_normal(sample)
    if sample["task"] == "translate" and sample and sample["language"] == "nn":
        return sample

def processor_normal_en(sample):
    sample = processor_normal(sample)
    if sample["task"] == "translate" and sample and sample["text_en"] is not None and sample["source"] != "audio_books":
        return {
            **sample,
            "text": sample["text_en"],
            "language": "en",
            "previous_text": None,
            "timestamped_text": None
        }

def processor_timestamps(sample):
    if sample["task"] == "translate" and sample["timestamped_text"] is not None:
        return {**sample, "text": sample["timestamped_text"], "previous_text": None}

def processor_timestamps_en(sample):
    if sample["task"] == "translate" and sample["timestamped_text_en"] is not None and sample["source"] != "audio_books":
        return {**sample, "text": sample["timestamped_text_en"], "language": "en","timestamped_text": sample["timestamped_text_en"], "previous_text": None}

def processor_previous_text_prompts(sample):
    if sample["task"] == "translate" and sample["previous_text"] is not None:
        return {**sample, "timestamped_text": None}

def processor_timestamps_previous_text(sample):
    if sample["task"] == "translate" and (sample["previous_text"] is not None
        and sample["timestamped_text"] is not None):
        return {**sample, "text": sample["timestamped_text"]}
  
def load_dataset_nbwhisper_rc_semantic(dataset_name, dataset_config_name=None, split="train", streaming=True, **kwargs):
    if split == "train":
        processors = [
            processor_normal_no,
            processor_normal_nn,
            processor_normal_en,
            processor_timestamps,
            processor_timestamps_en,
            processor_previous_text_prompts,
            processor_timestamps_previous_text,
        ]
    else:
        processors = processor_test
        
    ds = datasets.load_dataset(dataset_name, dataset_config_name, split=split, streaming=streaming, post_processors=processors, **kwargs)
    return ds