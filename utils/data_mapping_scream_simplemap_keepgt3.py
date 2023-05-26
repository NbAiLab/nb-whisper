from typing import Union
from datasets import Dataset, DatasetDict
import string

def make_style_tags(example):
    if example['verbosity'] <= 3:
        return {**example, 'remove_example': True}
    elif example['source'] == 'Fleurs':
        return {**example, 'prompt': '', 'remove_example': False}
    elif example['verbosity'] == 6:
        return {**example, 'prompt': '', 'remove_example': False}
    else:
        return {**example, 'prompt': '#', 'remove_example': False}

# Should now handle both datasets and dataset dictionaries
def map_data(data: Union[Dataset, DatasetDict]) -> Union[Dataset, DatasetDict]:
    if isinstance(data, dict):
        return {key: dataset.map(make_style_tags).filter(lambda x: not x['remove_example']) for key, dataset in data.items()}
    else:
        return data.map(make_style_tags).filter(lambda x: not x['remove_example'])
