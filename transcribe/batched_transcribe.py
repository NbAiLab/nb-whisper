import jax
import jax.numpy as jnp
from flax.jax_utils import replicate, unreplicate
from flax.training.common_utils import shard
from jax import device_get, pmap
from transformers import WhisperProcessor
from whisper_jax import FlaxWhisperForConditionalGeneration

MODEL_NAME = "openai/whisper-tiny"

def print_shapes(params, prefix=""):
    """Recursively print shapes for the nested params dictionaries."""
    for k, v in params.items():
        if isinstance(v, dict):
            print_shapes(v, prefix=f"{prefix}/{k}")
        else:
            print(f"{prefix}/{k}:", v.shape)

def main():
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = FlaxWhisperForConditionalGeneration.from_pretrained(MODEL_NAME, dtype=jnp.bfloat16, return_dict=False)
    params = model.params
    
    # Create a dummy batched_features for demonstration
    sample_rate = 16000
    dummy_audio = jnp.array([[[0.1] * sample_rate * 10]] * 4)  # 4 samples, each of 10 seconds
    individual_features = [
        processor(dummy_audio[i], sampling_rate=sample_rate, return_tensors="np").input_features
        for i in range(4)
    ]
    batched_features = jnp.stack(individual_features)
    
    print("\n=== DEBUG INFO ===")
    print("Shape of batched_features:", batched_features.shape)
    print("Shapes in params:")
    print_shapes(params)
    print("====================\n")
    
    def generate_fn(input_features):
        return model.generate(
            input_features, task="transcribe", return_timestamps=False, max_length=model.config.max_length,
        ).sequences
    
    p_generate = pmap(generate_fn, "input_features")
    params = replicate(params)
    
    pred_ids = p_generate(batched_features)
    output_ids = device_get(pred_ids.reshape(-1, model.config.max_length))
    
    transcription = processor.batch_decode(output_ids, skip_special_tokens=True)
    print("Transcription:", transcription)

if __name__ == "__main__":
    main()
