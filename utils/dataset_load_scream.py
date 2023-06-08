import random
import datasets


def processor_normal(sample):
    if sample["source"].upper() in ("NPSC", "NST", "FLEURS", "AUDIO BOOK"):
        return {**sample, "previous_text": None, "timestamped_text": None}


def processor_normal_no(sample):
    sample = processor_normal(sample)
    if sample and sample["text_language"] == "no":
        return sample


def processor_normal_nn(sample):
    sample = processor_normal(sample)
    if sample and sample["text_language"] == "nn":
        return sample


def processor_timestamps(sample):
    if sample["timestamped_text"] not in (None, ""):
        return {**sample, "text": sample["timestamped_text"], "previous_text": None}


def processor_previous_text_prompts(sample):
    if sample["previous_text"] is not None:
        return {**sample, "previous_text": sample["previous_text"], "timestamped_text": None}


def processor_style_prompts(sample):
    mapping = {
        "NRK TV": "[SUBTITLE]",
        "NRK TV TRANSLATE": "[SUBTITLE]",
        "NPSC": "[REPORT]",
        "NST": "[VERBATIM]",
        "FLEURS": "[VERBATIM]",
        "AUDIO BOOK": "[VERBATIM]",
    }
    return {**sample, "previous_text": mapping.get(sample["source"].upper(), ""), "timestamped_text": None}


def processor_previous_text_style_prompts(sample):
    mapping = {
        "NRK TV": "[SUBTITLE]",
        "NRK TV TRANSLATE": "[SUBTITLE]",
        "NPSC": "[REPORT]",
        "NST": "[VERBATIM]",
        "FLEURS": "[VERBATIM]",
        "AUDIO BOOK": "[VERBATIM]",
    }
    if sample["previous_text"] is not None:
        return {**sample, "previous_text": sample["previous_text"] + " " + mapping.get(sample["source"].upper(), ""), "timestamped_text": None}


def processor_timestamps_previous_text_style_prompts(sample):
    mapping = {
        "NRK TV": "[SUBTITLE]",
        "NRK TV TRANSLATE": "[SUBTITLE]",
        "NPSC": "[REPORT]",
        "NST": "[VERBATIM]",
        "FLEURS": "[VERBATIM]",
        "AUDIO BOOK": "[VERBATIM]",
    }
    if sample["previous_text"] is not None and sample["timestamped_text"] not in (None, ""):
        return {
            **sample,
            "text": sample["timestamped_text"],
            "previous_text": sample["previous_text"] + " " + mapping.get(sample["source"].upper(), ""),
        }


def processor_translate_to_english(sample):
    if sample["translated_text_en"] is not None:
        return {**sample, "text": sample["translated_text_en"], "task": "translate", "language": "no"}


def processor_en_transcribe(sample):
    if sample["translated_text_en"] is not None:
        return {**sample, "text": sample["translated_text_en"], "language": "en"}


def processor_nn_transcribe(sample):
    if sample["translated_text_nn"] is not None:
        return {**sample, "text": sample["translated_text_nn"], "language": "nn"}


def processor_es_transcribe(sample):
    if sample["translated_text_es"] is not None:
        return {**sample, "text": sample["translated_text_es"], "language": "es"}


def load_dataset_scream(dataset_name, dataset_config_name=None, split="train", streaming=True, **kwargs):
    if split == "train":
        processors = [
            processor_normal,
            processor_normal_no,
            processor_normal_nn,
            processor_timestamps,
            processor_previous_text_prompts,
            processor_style_prompts,
            processor_previous_text_style_prompts,
            processor_timestamps_previous_text_style_prompts,
            processor_translate_to_english,
            processor_en_transcribe,
            processor_nn_transcribe,
            processor_es_transcribe,
        ]
    else:
        processors = None
    ds = datasets.load_dataset(dataset_name, dataset_config_name, split=split, streaming=streaming, post_processors=processors, **kwargs)
    return ds
