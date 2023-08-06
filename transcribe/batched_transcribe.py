import jax.numpy as jnp
from datasets import load_dataset
from transformers import WhisperProcessor
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from jax import pmap, device_get
from whisper_jax import FlaxWhisperForConditionalGeneration

# Constants
DATASET = 'NbAiLab/ncc_speech_v3'
BATCH_SIZE = 4
MODEL_NAME = "openai/whisper-tiny"

def fetch_first_n_items(dataset, n):
    items = []
    iterator = iter(dataset)
    for _ in range(n):
        try:
            item = next(iterator)
            items.append(item)
        except StopIteration:
            break
    return items

def preprocess_audio(audio_items, processor):
    batched_features = []
    for item in audio_items:
        audio_data = item['audio']['array']
        sampling_rate = item['audio']['sampling_rate']
        features = processor(audio_data, return_tensors="np", sampling_rate=sampling_rate)
        batched_features.append(features.input_features)
    return batched_features

def batch_audio_features(features_list):
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

    # Load the model
    model, params = FlaxWhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME, dtype=jnp.bfloat16, _do_init=False
    )
    
    # Define a generate function for transcription
    def generate_fn(input_features):
        pred_ids = model.generate(
            input_features, task="transcribe", return_timestamps=False, 
            max_length=model.config.max_length, params=params
        )
        return pred_ids.sequences
    
    # Parallelize the generate function for data parallelism
    p_generate = pmap(generate_fn, "input_features")
    params = replicate(params)  # Replicate the model parameters

    # Print details about p_generate and params
    print("Details about p_generate:")
    print(f"Type: {type(p_generate)}")
    print("Details about params:")
    print(f"Type: {type(params)}")
    # TODO: Additional structure details can be added if necessary

    # TODO: Feed the batched_features into the p_generate function

if __name__ == "__main__":
    main()
