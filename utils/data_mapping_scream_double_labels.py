from typing import Union
from datasets import Dataset, DatasetDict
import string

def make_style_tags(batch):
    new_examples = []
    for i in range(len(batch['source'])):  # Here we iterate over the examples in the batch
        example = {key: value[i] for key, value in batch.items()}  # We reconstruct each example from the batch
        if example['source'] == 'NST':
            new_examples.append({**example, 'prompt': ''})
            new_examples.append({**example, 'prompt': '[READING]'})
        elif example['source'] == 'Fleurs':
            new_examples.append({**example, 'prompt': ''})
        elif example['source'] == 'NPSC':
            new_examples.append({**example, 'prompt': '[PROCEEDING]'})
        elif example['source'] == 'NRK TV':
            new_examples.append({**example, 'prompt': '[SUBTITLE]'})
        else:
            print("There is potentially an error in the dataset. Please check the example below:")
            print(example)
            new_examples.append({**example, 'prompt': ''})
    return {'source': [ex['source'] for ex in new_examples],
            'verbosity': [ex['verbosity'] for ex in new_examples],
            'prompt': [ex['prompt'] for ex in new_examples]}  # Please adjust this to include all fields present in your dataset


# Should now handle both datasets and dataset dictionaries
def map_data(data: Union[Dataset, DatasetDict]) -> Union[Dataset, DatasetDict]:
    # Check if this is a dataset or a dataset dictionary
    if isinstance(data, dict):
        return {key: dataset.map(make_style_tags, batched=True) for key, dataset in data.items()}
    else:
        return data.map(make_style_tags, batched=True)
