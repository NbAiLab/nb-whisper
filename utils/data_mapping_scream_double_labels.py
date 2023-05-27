from typing import Union
from datasets import Dataset, DatasetDict, concatenate_datasets

def make_style_tags(example):
    if example['source'] == 'NST':
        return [{'prompt': ''}, {'prompt': '[READING]'}]
    elif example['source'] == 'Fleurs':
        return [{'prompt': ''}]
    elif example['source'] == 'NPSC':
        return [{'prompt': '[PROCEEDING]'}]
    elif example['source'] == 'NRK TV':
        return [{'prompt': '[SUBTITLE]'}]
    else:
        print("There is potentially an error in the dataset. Please check the example below:")
        print(example)
        return [{'prompt': ''}]

def map_data(data: Union[Dataset, DatasetDict]) -> Union[Dataset, DatasetDict]:
    if isinstance(data, DatasetDict):
        return DatasetDict({key: concatenate_datasets([Dataset.from_dict(ex) for ex in dataset.map(make_style_tags, remove_columns=dataset.column_names)]) for key, dataset in data.items()})
    elif isinstance(data, Dataset):
        return concatenate_datasets([Dataset.from_dict(ex) for ex in data.map(make_style_tags, remove_columns=data.column_names)])
    else:
        raise TypeError(f"Unsupported data type: {type(data)}. This function expects a Dataset or DatasetDict.")
