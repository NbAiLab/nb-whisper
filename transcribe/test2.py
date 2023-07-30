import jax
import jax.numpy as jnp
from flax import jax_utils
from transformers import WhisperProcessor
from whisper_jax import FlaxWhisperForConditionalGeneration

# Load the model and replicate the parameters across devices
model, params = FlaxWhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-tiny.en",
    _do_init=False,
    dtype=jnp.bfloat16,
)
params = jax_utils.replicate(params)

# Load the WhisperProcessor
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")

# Define the function to generate predictions
def generate_fn(input_features):
    pred_ids = model.generate(input_features, params=params)
    return pred_ids.sequences

# Map the generate function to multiple devices
p_generate_fn = jax.pmap(generate_fn, "batch")

# Example of a single batch of audio data
batch = {"audio": {"array": my_audio_data, "sampling_rate": my_sampling_rate}}

# Preprocess the audio data
input_features = processor(
    batch["audio"]["array"], sampling_rate=batch["audio"]["sampling_rate"], return_tensors="np"
).input_features[0]

# Convert the input features to a batched array
input_features = jnp.array([input_features])

# Generate predictions
pred_ids = p_generate_fn(input_features)

# Decode the predicted token IDs to text
transcriptions = processor.batch_decode(pred_ids[0], skip_special_tokens=True)

print(transcriptions)
