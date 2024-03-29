#!/usr/bin/env bash
export TOKENIZERS_PARALLELISM=false
export CMALLOC_VERBOSE=0
export TCMALLOC_VERBOSE=0
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=10000000000
MODEL_NAME="NbAiLab/nb-whisper-medium-fine4-npsc-norm-nohes"
OUTPUT_DIR="./ModelD"
WANDB_PROJECT="nb-whisper-public-beta-transcription"
BATCH_SIZE=160
NUM_BEAMS=1
#BATCH_SIZE=32
#NUM_BEAMS=3
MAX_LABEL_LENGTH=256
LOGGING_STEPS=500  # or save steps


python run_pseudo_labelling_gcloud.py \
  --model_name_or_path $MODEL_NAME \
  --dataset_name "NbAiLab/ncc_speech_inference_v5y" \
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
  --push_to_hub False\
  --gcs_bucket "gs://nb-whisper-transcript" \
  --dataloader_num_workers 32 \
  --return_timestamps False \
  --language "Nynorsk" \
  --language_code "<|nn|>" \
  --task "transcribe" \
  --hub_token "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"