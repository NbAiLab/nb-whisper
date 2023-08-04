import argparse
import jax.numpy as jnp
from whisper_jax import FlaxWhisperPipline
from datasets import load_dataset

# Dataset constant
DATASET = 'NbAiLab/ncc_speech_v3'

def load_model(model_name):
    # Instantiate pipeline
    pipeline = FlaxWhisperPipline(model_name, dtype=jnp.bfloat16)
    return pipeline

def load_data(split):
    # Load streaming dataset
    dataset = load_dataset(DATASET, split=split, streaming=True)
    return dataset

def main(model, split):
    # Load model and data
    pipeline = load_model(model)
    dataset = load_data(split)

    # Transcribe each audio file in the dataset
    for i, item in enumerate(dataset):
        audio_file = item['audio']
        text = pipeline(audio_file, task="translate", language="Norwegian")
        print(f"Item {i} transcription: {text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transcribe audio from Huggingface streaming dataset')
    parser.add_argument('--model', type=str, default="openai/whisper-tiny-en", help='Model to use for transcription')
    parser.add_argument('--split', type=str, default="train", help='Split of the dataset to use')
    args = parser.parse_args()

    main(args.model, args.split)
