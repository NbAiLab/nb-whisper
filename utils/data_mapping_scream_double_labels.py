from typing import Union
from datasets import Dataset, DatasetDict
import string

def make_style_tags(batch):
    new_examples = []
    for example in batch:
        if example['source'] == 'NST':
            new_examples.append({**example, 'prompt': ''})
            new_examples.append({**example, 'prompt': '[READING]'})
        elif example['source'] == 'Fleurs':
            new_examples.append({**example, 'prompt': ''})
        elif example['verbosity'] <= 2 and example['source'] == 'NPSC':
            new_examples.append({**example, 'prompt': '[PROCEEDING]'})
        elif example['verbosity'] >= 3 and example['source'] == 'NRK TV':
            new_examples.append({**example, 'prompt': '[SUBTITLE]'})
        else:
            print("There is potentially an error in the dataset. Please check the example below:")
            print(example)
            new_examples.append({**example, 'prompt': ''})
    return new_examples

# Should now handle both datasets and dataset dictionaries
def map_data(data: Union[Dataset, DatasetDict]) -> Union[Dataset, DatasetDict]:
    # Check if this is a dataset or a dataset dictionary
    if isinstance(data, dict):
        return {key: dataset.map(make_style_tags, batched=True) for key, dataset in data.items()}
    else:
        return data.map(make_style_tags, batched=True)
