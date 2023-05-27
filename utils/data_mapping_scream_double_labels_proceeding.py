from typing import Union
from datasets import Dataset, DatasetDict
import string
import random

def make_style_tags(example):
    if example['source'] == 'NPSC':
        return {'prompt': ''} if random.random() < 0.5 else {'prompt': '[PROCEEDING]'}
    elif example['source'] == 'Fleurs':
        return {**example, 'prompt': ''}
    elif example['source'] == 'NST':
        return {**example, 'prompt': '[READING]'}
    elif example['source'] == 'NRK TV':
        return {**example, 'prompt': '[SUBTITLE]'}
    else:
        print("There is potentially an error in the dataset. Please check the example below:")
        print(example)
        return {**example, 'prompt': ''}

# Should now handle both datasets and dataset dictionaries
def map_data(data: Union[Dataset, DatasetDict]) -> Union[Dataset, DatasetDict]:
    # Check if this is a dataset or a dataset dictionary
    if isinstance(data, dict):
        return {key: dataset.map(make_style_tags) for key, dataset in data.items()}
    else:
        return data.map(make_style_tags)