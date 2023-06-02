import datasets


def load_default_transcribe(dataset_name, dataset_config_name, split, streaming=True, **kwargs):
    ds = datasets.load_dataset(dataset_name, dataset_config_name, split=split, streaming=streaming, **kwargs)  # .shuffle()
    ds = ds.cast_column("audio", datasets.features.Audio())
    ds = ds.filter(
        lambda sample: sample["verbosity"] > 3
    )
    ds = ds.map(
        lambda sample: {"task": "transcribe", "prompt": sample['previous_text'], **sample}
    )
    return ds

def load_subtitle_transcribe(dataset_name, dataset_config_name, split, streaming=True, **kwargs):
    ds = datasets.load_dataset(dataset_name, dataset_config_name, split=split, streaming=streaming, **kwargs)  # .shuffle()
    ds = ds.cast_column("audio", datasets.features.Audio())
    ds = ds.filter(
        lambda sample: sample["verbosity"] <= 3
    )
    ds = ds.map(
        lambda sample: {"task": "transcribe", "prompt": f"{sample['previous_text']} [SUBS]", **sample}
    )
    return ds

def load_dataset_scream(dataset_name, dataset_config_name=None, split="train", streaming=True, **kwargs):
    loaders = [
        load_default_transcribe,
        load_subtitle_transcribe
    ]
    return datasets.interleave_datasets(
        [loader(dataset_name, dataset_config_name, split, streaming=streaming, **kwargs) for loader in loaders],
        probabilities=[0.8, 0.2]
    )