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
GCS_BUCKET = "gs://nb-whisper-transcript"

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

    # Modify model name for output file
    model_name = model.split("/")[-1]

    # Check if output file exists
    output_file = f'output_{model_name}.txt'
    if os.path.exists(output_file):
        df = pd.read_csv(output_file, sep='\t')
    else:
        df = pd.DataFrame(columns=['id', 'target', model])

    # Create a set of already transcribed ids for faster lookup
    transcribed_ids = set(df['id'].values.tolist())

    # List of splits to load
    splits = ['train', 'validation', 'test'] if split is None else [split]

    # Control flag for max transcripts
    reached_max = False

    for split in splits:
        if reached_max:
            break

        dataset = load_data(split)

        # Transcribe each audio file in the dataset
        count = len(df)  # Start count from current DataFrame size
        start_time = time.time()
        try:
            for i, item in enumerate(dataset):
                if reached_max:
                    break

                if item['id'] not in transcribed_ids:  # Skip item if already transcribed
                    audio_file = item['audio']
                    result = pipeline(audio_file, task="transcribe", language="Norwegian")  # Changed task to "transcribe"
                    text = result['text']  # Extract text from result

                    # Add transcription to dataframe
                    new_row = pd.DataFrame({'id': [item['id']], 'target': [item['text']], model: [text]}, index=[count])
                    df = pd.concat([df, new_row])

                    transcribed_ids.add(item['id'])  # Add the transcribed id to the set
                    count += 1

                    # Push to output file every PUSH_INTERVAL steps
                    if count % PUSH_INTERVAL == 0:
                        elapsed_time = time.time() - start_time
                        transcription_speed = PUSH_INTERVAL / elapsed_time  # Calculate transcription speed
                        df.to_csv(output_file, sep='\t', index=False)  # Overwrite file
                        print(f'Saved {count} items to {output_file}. Transcription speed: {transcription_speed:.2f} items/second.')
                        
                        # Copy file to Google Cloud Storage
                        gcs_path = os.path.join(GCS_BUCKET, output_file)
                        copy_command = f"gsutil cp {output_file} {gcs_path}"
                        result = os.system(copy_command)
                        if result == 0:
                            print(f'Successfully copied {output_file} to {gcs_path}')
                        else:
                            print(f'Failed to copy {output_file} to {gcs_path}. Continuing transcription...')

                        start_time = time.time()

                # Exit gracefully if max transcripts is reached
                if count >= max:
                    print(f'Reached max transcripts: {max}')
                    reached_max = True
                    break
        except StopIteration:
            print(f"End of {split} split reached")

    # Save remaining transcripts
    df.to_csv(output_file, sep='\t', index=False)  # Overwrite file
    print(f'Saved {len(df)} items to {output_file}')

    # Copy file to Google Cloud Storage
    gcs_path = os.path.join(GCS_BUCKET, output_file)
    copy_command = f"gsutil cp {output_file} {gcs_path}"
    result = os.system(copy_command)
    if result == 0:
        print(f'Successfully copied {output_file} to {gcs_path}')
    else:
        print(f'Failed to copy {output_file} to {gcs_path}. Continuing transcription...')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transcribe audio from Huggingface streaming dataset')
    parser.add_argument('--model', type=str, default="openai/whisper-tiny-en", help='Model to use for transcription')
    parser.add_argument('--split', type=str, default=None, help='Split of the dataset to use')
    parser.add_argument('--max', type=int, default=float('inf'), help='Max number of transcripts')
    args = parser.parse_args()

    main(args.model, args.split, args.max)
