import jax.numpy as jnp
from datasets import load_dataset
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from jax import device_get, pmap
from transformers import WhisperProcessor
from whisper_jax import FlaxWhisperForConditionalGeneration

# load the processor and model
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
model, params = FlaxWhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v2", dtype=jnp.bfloat32, _do_init=False,
)

def generate_fn(input_features):
    pred_ids = model.generate(
        input_features, task="transcribe", return_timestamps=False, max_length=model.config.max_length, params=params,
    )
    return pred_ids.sequences

# pmap the generate function for data parallelism
p_generate = pmap(generate_fn, "input_features")
# replicate the parameters across devices
params = replicate(params)

# load a dummy sample from the LibriSpeech dataset
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = ds[0]["audio"]

# pre-process: convert the audio array to log-mel input features
# Assume input_features_single is a single example of shape (1, 80, 3000)
input_features_single = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="np").input_features.squeeze()

# Stack 4 identical copies to create a batch
input_features = jnp.stack([input_features_single] * 4)

# Make sure input features is of shape (4, 80, 3000)
print(input_features.shape)  # it should print: (4, 80, 3000)

# replicate the input features across devices for DP
input_features = shard(input_features)

# run the forward pass (JIT compiled the first time it is called)
pred_ids = p_generate(input_features)
output_ids = device_get(pred_ids.reshape(-1, model.config.max_length))

# post-process: convert tokens ids to text string
transcription = processor.batch_decode(output_ids, skip_special_tokens=True)
