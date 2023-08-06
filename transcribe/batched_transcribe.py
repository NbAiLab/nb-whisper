import jax.numpy as jnp
from datasets import load_dataset
from transformers import WhisperProcessor

# Constants
DATASET = 'NbAiLab/ncc_speech_v3'
BATCH_SIZE = 4
MODEL_NAME = "openai/whisper-tiny"  # Placeholder, replace with the appropriate model name if different

def fetch_first_n_items(dataset, n):
    items = []
    iterator = iter(dataset)  # Convert dataset into an iterator
    for _ in range(n):
        try:
            item = next(iterator)
            items.append(item)
        except StopIteration:
            break
    return items

def preprocess_audio(audio_items, processor):
    # Convert audio items to features using the WhisperProcessor
    batched_features = []
    for item in audio_items:
        audio_data = item['audio']
        features = processor(audio_data, return_tensors="jax")
        batched_features.append(features)
    return batched_features

def main():
    # Load dataset
    dataset = load_dataset(DATASET, split='train', streaming=True)
    
    # Fetch the first 4 audio items
    audio_items = fetch_first_n_items(dataset, BATCH_SIZE)
    
    # Initialize the WhisperProcessor
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    
    # Preprocess the audio items
    batched_features = preprocess_audio(audio_items, processor)
    
    # TODO: Feed the batched_features into the model and post-process the results

if __name__ == "__main__":
    main()
