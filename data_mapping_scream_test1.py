from typing import Union
from datasets import Dataset, DatasetDict
import string

def make_style_tags(example):
    if example['source'] == 'NPSC' or example['source'] == 'Fleurs' or example['source'] == 'NST':
        return {**example, 'prompt': ''}
    elif example['verbosity'] <= 2 and example['source'] == 'NRK TV':
        return {**example, 'prompt': '[rv]'}
    elif example['verbosity'] >= 3 and example['source'] == 'NRK TV':
        return {**example, 'prompt': '[nv]'}
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