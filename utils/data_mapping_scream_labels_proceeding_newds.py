from typing import Union
from datasets import Dataset, DatasetDict
import string

def make_style_tags(example):
    if example['source'] == 'stortinget' or example['source'] == 'audio_books' or example['source'] == 'nrk_tv_silence' or example['source'] == 'fleurs':
        return {**example, 'prompt': ''}
    elif example['source'] == 'nst':
        return {**example, 'prompt': '[verbatim]'}
    elif example['source'] == 'nrk_tv' or example['source'] == 'nrk_tv_translate':
        return {**example, 'prompt': '[subtitle]'}
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