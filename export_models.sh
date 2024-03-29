#!/bin/bash
pip install "optimum[exporters]>=1.14.1" tensorflow
git lfs track *.onnx*

python << END
from transformers import WhisperForConditionalGeneration, TFWhisperForConditionalGeneration, WhisperTokenizerFast
import shutil

# Backup generation_config.json - this is for tensorflow only, but at the moment that is causing errors.
# shutil.copyfile('./generation_config.json', './generation_config_backup.json')

print("Saving model to PyTorch...", end=" ")
model = WhisperForConditionalGeneration.from_pretrained("./", from_flax=True)
model.save_pretrained("./", safe_serialization=True)
model.save_pretrained("./", safe_serialization=False, max_shard_size="10000MB")
print("Done.")

#print("Saving model to TensorFlow...", end=" ")
#tf_model = TFWhisperForConditionalGeneration.from_pretrained("./")
#tf_model.save_pretrained("./")
#print("Done.")

# Restore the backup of generation_config.json
#shutil.move('./generation_config_backup.json', './generation_config.json')

print("Saving model to ONNX...", end=" ")
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
ort_model = ORTModelForSpeechSeq2Seq.from_pretrained("./", export=True)
ort_model.save_pretrained("./onnx")
print("Done")

END

echo "Saving model to CTranslate..."
ct2-transformers-converter --model . --output_dir ct2 --force
cp ct2/model.bin .
cp ct2/vocabulary.json .
cp config.json config_hf.json
jq -s '.[0] * .[1]' ct2/config.json config_hf.json > config.json
echo "Done"


echo "Saving model to GGML (whisper.cpp)..."
wget -O convert-h5-to-ggml.py "https://raw.githubusercontent.com/NbAiLab/nb-whisper/main/convert-h5-to-ggml.py"
mkdir -p whisper/assets
wget -O whisper/assets/mel_filters.npz "https://github.com/openai/whisper/raw/c5d42560760a05584c1c79546a098287e5a771eb/whisper/assets/mel_filters.npz"
python ./convert-h5-to-ggml.py ./ ./ ./
rm ./convert-h5-to-ggml.py
rm -rf ./whisper
echo "Done"

echo "Quantizing GGML model..."
git clone --depth 1 https://github.com/ggerganov/whisper.cpp --branch v1.5.1
cd whisper.cpp/
make -j 32
make quantize -j 32
./quantize ../ggml-model.bin ../ggml-model-q5_0.bin q5_0
cd ..
rm -rf whisper.cpp
echo "Done"
