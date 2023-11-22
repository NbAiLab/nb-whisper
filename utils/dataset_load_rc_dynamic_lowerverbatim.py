import random
import datasets
import string
def remove_punctuation_and_lowercase(s):
    return s.translate(str.maketrans('', '', string.punctuation)).lower()

def processor_test(sample):
    if sample["source"] == "nrk_tv" or sample["source"] == "nrk_tv_silence" or sample["source"] == "nrk_translate" or sample["source"] == "stortinget":
        return {**sample, "previous_text": None, "timestamped_text": None, "task": "translate"}
    else:
        return {**sample, "previous_text": None, "timestamped_text": None, "task": "transcribe"}

def processor_test_verbatim(sample):
    if sample:
        return {**sample, "previous_text": None, "timestamped_text": None, "task": "transcribe"}

def processor_test_semantic(sample):
    if sample:
        return {**sample, "previous_text": None, "timestamped_text": None, "task": "translate"}
    
def processor_normal(sample):
    return {**sample, "previous_text": None, "timestamped_text": None}

def processor_normal_no(sample):
    sample = processor_normal(sample)
    if sample and sample["language"] == "no":
        if sample["task"] == "transcribe":
            return {**sample, "text": remove_punctuation_and_lowercase(sample["text"])}
        else:
            return sample

def processor_normal_nn(sample):
    sample = processor_normal(sample)
    if sample and sample["language"] == "nn":
        if sample["task"] == "transcribe":
            return {**sample, "text": remove_punctuation_and_lowercase(sample["text"])}
        else:
            return sample

def processor_normal_en(sample):
    sample = processor_normal(sample)
    if sample and sample["text_en"] is not None and sample["source"] != "audio_books":
        return {
            **sample,
            "text": remove_punctuation_and_lowercase(sample["text_en"]),
            "language": "en",
            "previous_text": None,
            "timestamped_text": None
        }

def processor_timestamps(sample):
    if sample["timestamped_text"] is not None:
        if sample["task"] == "transcribe":
            return {**sample, "text": remove_punctuation_and_lowercase(sample["timestamped_text"]), "previous_text": None}
        else:
            return {**sample, "text": sample["timestamped_text"], "previous_text": None}

        

def processor_timestamps_en(sample):
    if sample["timestamped_text_en"] is not None and sample["source"] != "audio_books":
        if sample["task"] == "transcribe":
            return {**sample, "text": remove_punctuation_and_lowercase(sample["timestamped_text_en"]), "language": "en","timestamped_text": remove_punctuation_and_lowercase(sample["timestamped_text_en"]), "previous_text": None}
        else:
            return {**sample, "text": sample["timestamped_text_en"], "language": "en","timestamped_text": sample["timestamped_text_en"], "previous_text": None}

def processor_previous_text_prompts(sample):
    if sample["previous_text"] is not None:
        if sample["task"] == "transcribe":
            return {**sample, "timestamped_text": None, "text": remove_punctuation_and_lowercase(sample["text"])}
        else:
            return {**sample, "timestamped_text": None}


def processor_timestamps_previous_text(sample):
    if sample["previous_text"] is not None and sample["timestamped_text"] is not None:
        if sample["task"] == "transcribe":
            return {**sample, "text": sample["timestamped_text"], "text": remove_punctuation_and_lowercase(sample["text"])}
        else:
            return {**sample, "text": sample["timestamped_text"]}
        
  
def load_dataset_nbwhisper_rc_dynamic_lowerverbatim(dataset_name, dataset_config_name=None, split="train", streaming=True, **kwargs):
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

def load_dataset_nbwhisper_rc_dynamic_eval_verbatim_lowerverbatim(dataset_name, dataset_config_name=None, split="train", streaming=True, **kwargs):
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
        processors = processor_test_verbatim
        
    ds = datasets.load_dataset(dataset_name, dataset_config_name, split=split, streaming=streaming, post_processors=processors, **kwargs)
    return ds

def load_dataset_nbwhisper_rc_dynamic_eval_semantic_lowerverbatim(dataset_name, dataset_config_name=None, split="train", streaming=True, **kwargs):
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
        processors = processor_test_semantic
        
    ds = datasets.load_dataset(dataset_name, dataset_config_name, split=split, streaming=streaming, post_processors=processors, **kwargs)
    return ds