import argparse
import os
import time
import jax.numpy as jnp
import pandas as pd
from whisper_jax import FlaxWhisperPipline
from datasets import load_dataset

# Constants
DATASET = 'NbAiLab/ncc_speech_v3'
PUSH_INTERVAL = 50

def load_model(model_name):
    # Instantiate pipeline
    pipeline = FlaxWhisperPipline(model_name, dtype=jnp.bfloat16)
    return pipeline

def load_data(split):
    # Load streaming dataset
    dataset = load_dataset(DATASET, split=split, streaming=True)
    return dataset

def main(model, split, max):
    # Load model and data
    pipeline = load_model(model)
    dataset = load_data(split)

    # Modify model name for output file
    model_name = model.split("/")[-1]

    # Check if output file exists
    output_file = f'output_{model_name}.txt'
    if os.path.exists(output_file):
        df = pd.read_csv(output_file, sep='\t')
    else:
        df = pd.DataFrame(columns=['id', 'target', model])

    # Transcribe each audio file in the dataset
    count = 0
    start_time = time.time()
    try:
        for i, item in enumerate(dataset):
            if item['id'] not in df['id'].values:  # Skip item if already transcribed
                audio_file = item['audio']
                text = pipeline(audio_file, task="translate", language="Norwegian")

                # Add transcription to dataframe
                df.loc[count] = [item['id'], item['text'], text]

                count += 1

                # Push to output file every PUSH_INTERVAL steps
                if count % PUSH_INTERVAL == 0:
                    elapsed_time = time.time() - start_time
                    df.to_csv(output_file, sep='\t', index=False)
                    print(f'Saved {count} items to {output_file}. Transcription speed: {PUSH_INTERVAL / elapsed_time} items/second.')
                    start_time = time.time()

            # Exit gracefully if max transcripts is reached
            if count >= max:
                print(f'Reached max transcripts: {max}')
                break
    except StopIteration:
        print("End of dataset reached")

    # Save remaining transcripts
    df.to_csv(output_file, sep='\t', index=False)
    print(f'Saved {count} items to {output_file}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transcribe audio from Huggingface streaming dataset')
    parser.add_argument('--model', type=str, default="openai/whisper-tiny-en", help='Model to use for transcription')
    parser.add_argument('--split', type=str, default="train", help='Split of the dataset to use')
    parser.add_argument('--max', type=int, default=float('inf'), help='Max number of transcripts')
    args = parser.parse_args()

    main(args.model, args.split, args.max)
