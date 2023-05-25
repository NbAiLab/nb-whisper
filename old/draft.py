from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torchaudio



model_name = "NbAiLab/scream_tertius_simplemap";processor = WhisperProcessor.from_pretrained(model_name, cache_dir="prompt"); model = WhisperForConditionalGeneration.from_pretrained(model_name, cache_dir="prompt", from_flax=True)
input_speech, sr = torchaudio.load("audio6.mp3");input_features = processor(input_speech.squeeze(), sampling_rate=sr, return_tensors="pt", from_flax=True).input_features

# --- Without prompt ---
output_without_prompt = model.generate(input_features, language="no");print(processor.decode(output_without_prompt[0], skip_special_tokens=False))


# --- With prompt ---
prompt_ids = processor.get_prompt_ids("[rv]");output_with_prompt = model.generate(input_features, prompt_ids=prompt_ids, language="no"); print(processor.decode(output_with_prompt[0], skip_special_tokens=False))

