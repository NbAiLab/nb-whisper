from datasets import Dataset

def transform_text_to_xxx(dataset: Dataset) -> Dataset:
    return dataset.map(lambda example: {'text': "xxx"})

def add_yyy_to_source(dataset: Dataset) -> Dataset:
    return dataset.map(lambda example: {'source': example['source'] + " yyy"})

def map_data(dataset: Dataset) -> Dataset:
    dataset = transform_text_to_xxx(dataset)
    dataset = add_yyy_to_source(dataset)
    return dataset