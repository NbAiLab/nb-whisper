from typing import Union
from datasets import Dataset, DatasetDict

def transform_text_to_xxx(dataset: Dataset) -> Dataset:
    return dataset.map(lambda example: {'text': "xxx"})

def add_yyy_to_source(dataset: Dataset) -> Dataset:
    return dataset.map(lambda example: {'source': example['source'] + " yyy"})

# Should now handle both datasets and dataset dictionaries
def map_data(data: Union[Dataset, DatasetDict]) -> Union[Dataset, DatasetDict]:
    
    # Do the mapping
    def transform(dataset: Dataset) -> Dataset:
        dataset = transform_text_to_xxx(dataset)
        dataset = add_yyy_to_source(dataset)
        return dataset

    # Check if this is a dataset or a dataset dictionary
    if isinstance(data, dict):
        return {key: transform(dataset) for key, dataset in data.items()}
    else:
        return transform(data)