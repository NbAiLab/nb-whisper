#!/usr/bin/env bash

MODEL_NAME="openai/whisper-large-v2"
OUTPUT_DIR="./transcriptions-streaming"
WANDB_PROJECT="nb-whisper-public-beta"
BATCH_SIZE=64
NUM_BEAMS=1
MAX_LABEL_LENGTH=256
LOGGING_STEPS=500  # or save steps

python run_pseudo_labelling.py \
  --model_name_or_path $MODEL_NAME \
  --dataset_name "speechcolab/gigaspeech" \
  --dataset_config_name "xs" \
  --data_split_name "train+validation+test" \
  --text_column_name "text" \
  --wandb_name "nb-whisper-transcribe" \
  --id_column_name "segment_id" \
  --dataset_cache_dir $CACHE_DIR \
  --output_dir $OUTPUT_DIR \
  --wandb_project $WANDB_PROJECT \
  --per_device_eval_batch_size $BATCH_SIZE \
  --generation_num_beams $NUM_BEAMS \
  --max_label_length $MAX_LABEL_LENGTH \
  --logging_steps $LOGGING_STEPS \
  --dtype "bfloat16" \
  --report_to "wandb" \
  --streaming \
  --push_to_hub \
  --dataloader_num_workers 32 \
  --hub_token "YOUR_TOKEN_HERE"