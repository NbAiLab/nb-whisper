from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torchaudio

def print_predictions():
    for a in ["audio1.mp3","audio2.mp3","audio3.mp3","audio4.mp3","audio5.mp3","audio6.mp3","audio7.mp3"]:
        print(f"\n Predicting {a} 1) [READING], 2) [SUBTITLE], 3) NOPROMPT")
        input_speech, sr = torchaudio.load(a);input_features = processor(input_speech.squeeze(), sampling_rate=sr, return_tensors="pt", from_flax=True).input_features
        prompt_ids = processor.get_prompt_ids("[READING]");output_with_prompt = model.generate(input_features, prompt_ids=prompt_ids, language="no"); print("1)" + processor.decode(output_with_prompt[0], skip_special_tokens=True))
        prompt_ids = processor.get_prompt_ids("[SUBTITLE]");output_with_prompt = model.generate(input_features, prompt_ids=prompt_ids, language="no"); print("2)" + processor.decode(output_with_prompt[0], skip_special_tokens=True))
        output_without_prompt = model.generate(input_features, language="no");print("3)" + processor.decode(output_without_prompt[0], skip_special_tokens=True))


model_name = "NbAiLab/scream_tertius_simplemap_labels_proceeding_nominus100_bpedropout";processor = WhisperProcessor.from_pretrained(model_name, cache_dir="prompt"); model = WhisperForConditionalGeneration.from_pretrained(model_name, cache_dir="prompt", from_flax=True)
print_predictions()


input_speech, sr = torchaudio.load("audio6.mp3");input_features = processor(input_speech.squeeze(), sampling_rate=sr, return_tensors="pt", from_flax=True).input_features

# --- Without prompt ---
output_without_prompt = model.generate(input_features, language="no");print(processor.decode(output_without_prompt[0], skip_special_tokens=False))


# --- With prompt ---
prompt_ids = processor.get_prompt_ids("[rv]");output_with_prompt = model.generate(input_features, prompt_ids=prompt_ids, language="no"); print(processor.decode(output_with_prompt[0], skip_special_tokens=False))




