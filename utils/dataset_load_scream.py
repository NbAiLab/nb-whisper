import datasets


def load_normal(ds):
    ds = ds.filter(lambda sample: sample["source"].upper() in ("NPSC", "NST", "FLEURS", "AUDIO BOOK"))
    return ds


def load_normal_no(ds):
    ds = load_normal(ds)
    ds = ds.filter(lambda sample: sample["text_language"] == "no")
    return ds


def load_normal_nn(ds):
    ds = load_normal(ds)
    ds = ds.filter(lambda sample: sample["text_language"] == "nn")
    return ds


def load_timestamps(ds):
    ds = ds.filter(lambda sample: sample["timestamped_text"] is not None)
    ds = ds.map(lambda sample: {**sample, "text": sample["timestamped_text"], "has_timestamps": True})
    return ds


def load_previous_text_prompts(ds):
    ds = ds.filter(lambda sample: sample["previous_text"] is not None)
    ds = ds.map(lambda sample: {**sample, "prompt": sample["previous_text"]})
    return ds


def load_style_prompts(ds):
    mapping = {
        "NRK TV": "[SUBTITLE]",
        "NRK TV TRANSLATE": "[SUBTITLE]",
        "NPSC": "[REPORT]",
        "NST": "[TEXT]",
        "FLEURS": "[TEXT]",
        "AUDIO BOOK": "[TEXT]",
    }
    # ds = ds.filter()
    ds = ds.map(lambda sample: {**sample, "prompt": mapping.get(sample["source"].upper(), "")})
    return ds


def load_translate_to_english(ds):
    ds = ds.filter(lambda sample: sample["translated_text_en"] is not None)
    ds = ds.map(lambda sample: {**sample, "text": sample["translated_text_en"], "task": "translate", "language": "no"})
    return ds


def load_en_transcribe(ds):
     ds = ds.filter(lambda sample: sample["translated_text_en"] is not None)
     ds = ds.map(lambda sample: {**sample, "text": sample["translated_text_en"], "language": "en"})


def load_nn_transcribe(ds):
     ds = ds.filter(lambda sample: sample["translated_text_nn"] is not None)
     ds = ds.map(lambda sample: {**sample, "text": sample["translated_text_nn"], "language": "nn"})


def load_es_transcribe(ds):
     ds = ds.filter(lambda sample: sample["translated_text_es"] is not None)
     ds = ds.map(lambda sample: {**sample, "text": sample["translated_text_es"], "language": "es"})


def load_dataset_scream(dataset_name, dataset_config_name=None, split="train", streaming=True, **kwargs):
    loaders = [
        (load_normal, .31),
        (load_normal_no, .1),
        (load_normal_nn, .1),
        (load_timestamps, .1),
        (load_previous_text_prompts, .1),
        (load_style_prompts, .1),
        # (load_translate_to_english, .1),
        # (load_en_transcribe, 0.03),
        # (load_nn_transcribe, 0.03),
        # (load_es_transcribe, 0.03),
    ]
    return datasets.interleave_datasets(
        [loader(
            datasets.load_dataset(dataset_name, dataset_config_name, split=split, streaming=streaming, **kwargs)
        ) for loader, _ in loaders],
        probabilities=[probability for _, probability in loaders],
    )