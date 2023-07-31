import time

import jax.numpy as jnp
from datasets import concatenate_datasets, load_dataset
from whisper_jax import FlaxWhisperPipline


BATCH_SIZES = [4, 8, 16, 32, 64, 128]
NUM_BATCHES = 100

# load a dataset of 73 audio samples
librispeech = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

for batch_size in BATCH_SIZES:
    eval_dataset = librispeech.select(range(batch_size // 2))
    eval_dataset = concatenate_datasets([eval_dataset for _ in range(2 * NUM_BATCHES)])

    eval_dataloader = eval_dataset.with_format("numpy").iter(batch_size=batch_size)

    # Create the FlaxWhisperPipeline object
    pipeline = FlaxWhisperPipeline(
        model_name_or_path="openai/whisper-tiny.en",
        dtype=jnp.bfloat16,  # use bfloat16 precision
        batch_size=batch_size,  # enable batching
    )

    # warm-up step
    batch = next(iter(eval_dataloader))
    # Pass the audio file path to the pipeline
    transcription = pipeline(batch["audio"]["path"])

    start = time.time()
    for batch in eval_dataloader:
        # Pass the audio file path to the pipeline
        transcription = pipeline(batch["audio"]["path"])
    runtime = time.time() - start

    print(f"{batch_size}: {runtime:.06}")
