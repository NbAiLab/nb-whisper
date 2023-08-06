import jax.numpy as jnp
from datasets import load_dataset
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from jax import device_get, pmap
from transformers import WhisperProcessor
from whisper_jax import FlaxWhisperForConditionalGeneration

MODEL_NAME = "openai/whisper-tiny"

def main():
    # Load the processor and model
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model,params = FlaxWhisperForConditionalGeneration.from_pretrained(MODEL_NAME, dtype=jnp.bfloat16, _do_init=False)

    # Load a dummy sample from the LibriSpeech dataset
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    sample = ds[0]["audio"]

    # Generate individual features
    dummy_audio = [sample["array"] for _ in range(4)]
    sample_rate = sample["sampling_rate"]

    print("=== DEBUG INFO ===")
    print("Shape of dummy_audio[0]:", dummy_audio[0].shape)
    
    individual_features = [
        processor(dummy_audio[i], sampling_rate=sample_rate, return_tensors="np").input_features
        for i in range(4)
    ]
    for idx, feat in enumerate(individual_features):
        print(f"Shape of individual_features[{idx}]:", feat.shape)

    # Stack features for batching
    batched_features = jnp.stack(individual_features)
    print("Shape of batched_features:", batched_features.shape)

    def generate_fn(input_features):
        pred_ids = model.generate(
            input_features, task="transcribe", return_timestamps=False, max_length=model.config.max_length, params=params['model'],
        )
        return pred_ids.sequences

    # pmap the generate function for data parallelism
    p_generate = pmap(generate_fn, "input_features")

    # replicate the parameters across devices
    params = replicate(params)

    
    # Run the forward pass (JIT compiled the first time it is called)
    
    # TODO : I am unable to run the line below. Keep getting the following error:
    # *** flax.errors.ScopeParamNotFoundError: Could not find parameter named "kernel" in scope "/model/encoder/conv1"
    # I get this even with batch 1. I have tried to debug this but I am unable to find the cause.
    breakpoint()
    pred_ids = p_generate(batched_features)
    output_ids = device_get(pred_ids.reshape(-1, model.config.max_length))

    # Post-process: convert tokens ids to text string
    transcription = processor.batch_decode(output_ids, skip_special_tokens=True)
    print("Transcription:", transcription)


if __name__ == "__main__":
    main()
