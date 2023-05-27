from typing import Union
from datasets import Dataset, DatasetDict
import pandas as pd

def make_style_tags(example):
    if example['source'] == 'NST':
        return [{**example, 'prompt': ''}, {**example, 'prompt': '[READING]'}]
    elif example['source'] == 'Fleurs':
        return [{**example, 'prompt': ''}]
    elif example['source'] == 'NPSC':
        return [{**example, 'prompt': '[PROCEEDING]'}]
    elif example['source'] == 'NRK TV':
        return [{**example, 'prompt': '[SUBTITLE]'}]
    else:
        print("There is potentially an error in the dataset. Please check the example below:")
        print(example)
        return [{**example, 'prompt': ''}]

def map_data(data: Union[Dataset, DatasetDict]) -> Union[Dataset, DatasetDict]:
    if isinstance(data, DatasetDict):
        return DatasetDict({key: Dataset.from_pandas(pd.DataFrame(make_style_tags(dataset))) for key, dataset in data.items()})
    elif isinstance(data, Dataset):
        return Dataset.from_pandas(pd.DataFrame(make_style_tags(data)))
    else:
        raise TypeError(f"Unsupported data type: {type(data)}. This function expects a Dataset or DatasetDict.")
