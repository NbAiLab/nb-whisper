import time
import jax
import jax.numpy as jnp
from datasets import concatenate_datasets, load_dataset
from flax import jax_utils
from flax.training.common_utils import shard
from transformers import FlaxWhisperForConditionalGeneration, WhisperProcessor

# Set your BATCH_SIZE according to your GPU's memory availability
BATCH_SIZE = 16
NUM_BATCHES = 5

# Load the model and replicate the parameters across devices
model, params = FlaxWhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-tiny-v2",
    _do_init=False,
    dtype=jnp.bfloat16,
)

params = jax_utils.replicate(params)

# Function to generate predictions
def generate_fn(batch):
    pred_ids = model.generate(batch, params=params)
    return pred_ids.sequences

# Map the generate function to multiple devices
p_generate_fn = jax.pmap(generate_fn, "batch")

# Load the WhisperProcessor
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")

# Function to preprocess the audio data
def preprocess(batch):
    batch["input_features"] = processor(
        batch["audio"]["array"], sampling_rate=batch["audio"]["sampling_rate"], return_tensors="np"
    ).input_features[0]
    return batch

# Provide your Hugging Face token here
audio_files_dataset = load_dataset("NbAiLab/ncc_speech_v3", split="train", use_auth_token=True, streaming=True)

# Preprocess the audio data
dataset_processed = audio_files_dataset.map(preprocess, remove_columns=audio_files_dataset.column_names)

# Create a dataloader
eval_dataloader = dataset_processed.with_format("numpy").iter(batch_size=BATCH_SIZE)

# Warm-up step
batch = next(iter(eval_dataloader))
input_features = shard(batch["input_features"])
pred_ids = p_generate_fn(input_features)

# Run the model on the batches
start = time.time()
for i, batch in enumerate(eval_dataloader):
    input_features = shard(batch["input_features"])
    pred_ids = p_generate_fn(input_features)
    
    # Post-process: convert tokens ids to text string
    transcriptions = processor.batch_decode(jax.device_get(pred_ids.reshape(-1, model.config.max_length)), skip_special_tokens=True)
    
    # Print the transcriptions
    for transcription in transcriptions:
        print(transcription)
    
    # Exit after processing NUM_BATCHES batches
    if i >= NUM_BATCHES - 1:
        break

runtime = time.time() - start
print(f"\nTotal time for {NUM_BATCHES} batches: {runtime:.06}")
