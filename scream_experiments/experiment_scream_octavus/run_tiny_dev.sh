export TOKENIZERS_PARALELLISM=false
export CMALLOC_VERBOSE=0
export TCMALLOC_VERBOSE=0
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=10000000000
python ../run_flax_speech_recognition_seq2seq_streaming_dev.py \
    --model_name_or_path openai/whisper-tiny \
    --run_name "ScreamTiny - debug" \
    --run_description "A Tiny Whisper Scream model - Debug delete." \
    --wandb_entity "nbailab" \
    --wandb_project "Scream - octavus" \
    --dataset_name NbAiLab/NCC_speech_v5_mini \
    --language Norwegian \
    --text_column_name text \
    --train_split_name train\
    --eval_split_name validation\
    --output_dir ../../scream_tiny_octavus_dev_mini\
    --overwrite_output_dir\
    --warmup_steps 1000 \
    --do_train \
    --do_eval \
    --num_train_steps 20000 \
    --lr_scheduler_type linear \
    --eval_steps 2000 \
    --learning_rate 3e-5 \
    --preprocessing_num_workers 32 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --predict_with_generate \
    --log_max_eval_predictions 100 \
    --log_eval_predictions_fn "log_predictions.write_predictions" \
    --streaming True \
    --use_auth_token True \
    --dtype bfloat16 \
    --hub_private_repo True \
    --hub_model_id NbAiLab/scream_tiny_octavus_dev_mini \
    --resume_from_checkpoint True \
    --ignore_data_skip True \
    --push_to_hub
