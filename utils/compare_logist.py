from datasets import load_dataset
from transformers import WhisperTokenizer, FlaxWhisperForConditionalGeneration, AutoFeatureExtractor, FlaxLogitsProcessor

from modeling_flax_whisper import FlaxWhisperForConditionalGeneration as ScanFlaxWhisperForConditionalGeneration


class ScoresFlaxLogitsProcessor(FlaxLogitsProcessor):

    def __call__(self, input_ids, scores, cur_len):
        print(scores)
        return scores

model = FlaxWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
scan_model = ScanFlaxWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
scan_model.enable_gradient_checkpointing()
scan_model.enable_scan()

feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-tiny")

ds = load_dataset("NbAiLab/NCC_speech_nrk_v5", streaming=True, split="test")
sample = next(iter(ds))
inputs = feature_extractor(sample["audio"]["array"], sampling_rate=sample["audio"]["sampling_rate"], return_tensors="jax")

print("Flax model")
model.generate(inputs["input_features"], logits_processor=[ScoresFlaxLogitsProcessor()], trace=False)
print("Flax model scan")
scan_model.generate(inputs["input_features"], logits_processor=[ScoresFlaxLogitsProcessor()], trace=False)
