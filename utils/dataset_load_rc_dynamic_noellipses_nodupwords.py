import random
import datasets
import re

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


import re

def filtered_wrapper(processor_function):
    """
    Wrapper function that applies various filters to samples with task 'translate' after the primary processor function.
    """
    def is_duplicate_word_present(text):
        # Using regex to find words consisting of the specified character range
        words = re.findall(r'\b[A-Za-zæøåÆØÅ]+\b', text)
        return len(words) != len(set(words))

    def wrapped_processor(sample):
        processed_sample = processor_function(sample)
        if processed_sample is None:
            return None

        if processed_sample.get("task") == "translate":
            if processed_sample.get("source") in ["nst", "audio_books"]:
                return None

            text = processed_sample.get("text", "")
            timestamped_text = processed_sample.get("timestamped_text", "")

            # Checking for duplicate words in text and timestamped_text
            if text and is_duplicate_word_present(text):
                return None
            if timestamped_text and is_duplicate_word_present(timestamped_text):
                return None

            # Ensure text and timestamped_text are not None before checking for "…"
            if text is not None and "…" in text:
                return None
            if timestamped_text is not None and "…" in timestamped_text:
                return None
        
        return processed_sample
    return wrapped_processor



def processor_normal(sample):
    return {**sample, "previous_text": None, "timestamped_text": None}

def processor_normal_no(sample):
    sample = processor_normal(sample)
    if sample and sample["language"] == "no":
        return sample

def processor_normal_nn(sample):
    sample = processor_normal(sample)
    if sample and sample["language"] == "nn":
        return sample

def processor_normal_en(sample):
    sample = processor_normal(sample)
    if sample and sample["text_en"] is not None and sample["source"] != "audio_books":
        return {
            **sample,
            "text": sample["text_en"],
            "language": "en",
            "previous_text": None,
            "timestamped_text": None
        }

def processor_timestamps(sample):
    if sample["timestamped_text"] is not None:
        return {**sample, "text": sample["timestamped_text"], "previous_text": None}

def processor_timestamps_en(sample):
    if sample["timestamped_text_en"] is not None and sample["source"] != "audio_books":
        return {**sample, "text": sample["timestamped_text_en"], "language": "en","timestamped_text": sample["timestamped_text_en"], "previous_text": None}

def processor_previous_text_prompts(sample):
    if sample["previous_text"] is not None:
        return {**sample, "timestamped_text": None}

def processor_timestamps_previous_text(sample):
    if sample["previous_text"] is not None and sample["timestamped_text"] is not None:
        return {**sample, "text": sample["timestamped_text"]}
  
def load_dataset_nbwhisper_rc_dynamic(dataset_name, dataset_config_name=None, split="train", streaming=True, **kwargs):
    processors = [
        processor_normal_no,
        processor_normal_nn,
        processor_normal_en,
        processor_timestamps,
        processor_timestamps_en,
        processor_previous_text_prompts,
        processor_timestamps_previous_text,
    ]

    if split == "train":
        processors = [filtered_wrapper(proc) for proc in processors]
    else:
        processors = [processor_test]
        
    ds = datasets.load_dataset(dataset_name, dataset_config_name, split=split, streaming=streaming, post_processors=processors, **kwargs)
    return ds

def load_dataset_nbwhisper_rc_dynamic_eval_verbatim(dataset_name, dataset_config_name=None, split="train", streaming=True, **kwargs):
    processors = [
        processor_normal_no,
        processor_normal_nn,
        processor_normal_en,
        processor_timestamps,
        processor_timestamps_en,
        processor_previous_text_prompts,
        processor_timestamps_previous_text,
    ]

    if split == "train":
        processors = [filtered_wrapper(proc) for proc in processors]
    else:
        processors = [processor_test_verbatim]
        
    ds = datasets.load_dataset(dataset_name, dataset_config_name, split=split, streaming=streaming, post_processors=processors, **kwargs)
    return ds

def load_dataset_nbwhisper_rc_dynamic_eval_semantic(dataset_name, dataset_config_name=None, split="train", streaming=True, **kwargs):
    processors = [
        processor_normal_no,
        processor_normal_nn,
        processor_normal_en,
        processor_timestamps,
        processor_timestamps_en,
        processor_previous_text_prompts,
        processor_timestamps_previous_text,
    ]

    if split == "train":
        processors = [filtered_wrapper(proc) for proc in processors]
    else:
        processors = [processor_test_semantic]
        
    ds = datasets.load_dataset(dataset_name, dataset_config_name, split=split, streaming=streaming, post_processors=processors, **kwargs)
    return ds