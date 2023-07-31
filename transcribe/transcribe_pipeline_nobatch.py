import time

import jax.numpy as jnp
from datasets import concatenate_datasets, load_dataset
from whisper_jax import FlaxWhisperPipline

# load a dataset of 73 audio samples
librispeech = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")


eval_dataloader = librispeech.with_format("numpy").iter(batch_size=4)

# Create the FlaxWhisperPipeline object
pipeline = FlaxWhisperPipline(
    "openai/whisper-tiny.en",
    dtype=jnp.bfloat16,  # use bfloat16 precision
    batch_size=4,  # enable batching
)

while True:
    audio = next(iter(eval_dataloader))
    start = time.time()
    transcription = pipeline(audio["audio"][0]["path"])
    print(audio["audio"][0]["path"])
    print(transcription)
    runtime = time.time() - start
    print(f"{runtime:.06}")

