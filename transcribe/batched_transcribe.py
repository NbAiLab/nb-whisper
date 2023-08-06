import jax.numpy as jnp
from datasets import load_dataset
from transformers import WhisperProcessor

# Constants
DATASET = 'NbAiLab/ncc_speech_v3'
BATCH_SIZE = 4
MODEL_NAME = "openai/whisper-tiny"

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
        audio_data = item['audio']['array']  # Extract the raw audio data from the dictionary
        sampling_rate = item['audio']['sampling_rate']
        features = processor(audio_data, return_tensors="np", sampling_rate=sampling_rate)
        batched_features.append(features.input_features)
    return batched_features

def batch_audio_features(features_list):
    # Combine the individual feature items into a single batch
    batch = jnp.concatenate(features_list, axis=0)
    return batch

def main():
    # Load dataset
    dataset = load_dataset(DATASET, split='train', streaming=True)
    
    # Fetch the first 4 audio items
    audio_items = fetch_first_n_items(dataset, BATCH_SIZE)
    
    # Initialize the WhisperProcessor
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    
    # Preprocess the audio items
    individual_features = preprocess_audio(audio_items, processor)
    
    # Combine individual features into a batch
    batched_features = batch_audio_features(individual_features)
    
    # Print details about the batched_features
    print("Details about batched_features:")
    print(f"Type: {type(batched_features)}")
    print(f"Shape: {batched_features.shape}")

if __name__ == "__main__":
    main()
