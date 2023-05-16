from typing import Union
from datasets import Dataset, DatasetDict
import string

def make_style_tags(dataset: Dataset) -> Dataset:
    def transform(example):
        if example['source'] == 'NPSC' or example['source'] == 'Fleurs':
            return {'prompt': ''}
        elif example['verbosity'] <= 2 and example['source'] == 'NRK TV':
            return {'prompt': '[rv]'}
        elif example['verbosity'] >= 3 and example['source'] == 'NRK TV':
            return {'prompt': '[nv]'}
        elif example['source'] == 'NST':
            text = example['text'].translate(str.maketrans('', '', string.punctuation)).lower()
            return {'prompt': f'[fv]', 'text': f'[{text}]'}
        else:
            print("There is potentially an error in the dataset. Please check the example below:")
            print(example)
            return {'prompt': ''}
    
    return dataset.map(transform)

# Should now handle both datasets and dataset dictionaries
def map_data(data: Union[Dataset, DatasetDict]) -> Union[Dataset, DatasetDict]:
    # Do the mapping
    def transform(dataset: Dataset) -> Dataset:
        dataset = make_style_tags(dataset)
        return dataset

    # Check if this is a dataset or a dataset dictionary
    if isinstance(data, dict):
        return {key: transform(dataset) for key, dataset in data.items()}
    else:
        return transform(data)