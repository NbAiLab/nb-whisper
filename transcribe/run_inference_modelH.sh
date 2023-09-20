#!/usr/bin/env bash

MODEL_NAME="https://huggingface.co/openai/whisper-medium"
OUTPUT_DIR="./ModelH"
WANDB_PROJECT="nb-whisper-public-beta-transcription"
BATCH_SIZE=80
NUM_BEAMS=1
#BATCH_SIZE=32
#NUM_BEAMS=3
MAX_LABEL_LENGTH=256
LOGGING_STEPS=100000  # or save steps


python run_pseudo_labelling.py \
  --model_name_or_path $MODEL_NAME \
  --dataset_name "NbAiLab/ncc_speech_v3" \
  --dataset_config_name "" \
  --data_split_name "train" \
  --text_column_name "text" \
  --id_column_name "id" \
  --output_dir $OUTPUT_DIR \
  --wandb_project $WANDB_PROJECT \
  --per_device_eval_batch_size $BATCH_SIZE \
  --generation_num_beams $NUM_BEAMS \
  --max_label_length $MAX_LABEL_LENGTH \
  --logging_steps $LOGGING_STEPS \
  --dtype "bfloat16" \
  --streaming \
  --push_to_hub \
  --dataloader_num_workers 32 \
  --return_timestamps True \
  --language "English" \
  --language_code "<|en|>" \
  --task "translate" \
  --hub_token "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"