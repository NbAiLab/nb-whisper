from typing import Union
from datasets import Dataset, DatasetDict
import string


def make_style_tags(example):
    if example['verbosity'] <= 3:
        return None
    elif example['source'] == 'Fleurs':
        return {**example, 'prompt': ''}
    elif example['verbosity'] == 6:
        return {**example, 'prompt': ''}
    else:
        return {**example, 'prompt': '#'}

# Should now handle both datasets and dataset dictionaries
def map_data(data: Union[Dataset, DatasetDict]) -> Union[Dataset, DatasetDict]:
    # Check if this is a dataset or a dataset dictionary
    if isinstance(data, dict):
        return {key: dataset.map(make_style_tags).filter(lambda x: x is not None) for key, dataset in data.items()}
    else:
        return data.map(make_style_tags).filter(lambda x: x is not None)