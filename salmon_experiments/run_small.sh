# Environment variables
export TOKENIZERS_PARALLELISM=false
export CMALLOC_VERBOSE=0
export TCMALLOC_VERBOSE=0
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=10000000000
export MODEL_HUB_ID=whisper-small-smj

# Running the Python script
python ../run_nb_flax_speech_recognition_seq2seq_streaming_dev.py \
    --model_name_or_path openai/whisper-small \
    --run_name "Salmon Whisper - small" \
    --run_description "A small Whisper for Lule Sámi" \
    --wandb_entity "nbailab" \
    --wandb_project "salmon" \
    --dataset_name NbAiLab/salmon-asr-smj \
    --language Estonian \
    --task transcribe \
    --text_column_name transcription \
    --train_split_name train \
    --test_split_name "test" \
    --eval_split_name "validation" \
    --hub_model_id NbAiLab/$MODEL_HUB_ID \
    --output_dir ../../$MODEL_HUB_ID \
    --overwrite_output_dir \
    --do_train \
    --do_predict \
    --do_eval \
    --predict_with_generate \
    --warmup_steps 1000 \
    --num_train_steps 10000 \
    --eval_steps 1000 \
    --lr_scheduler_type linear \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-6 \
    --bpe_dropout 0.2 \
    --activation_dropout 0.1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 8 \
    --preprocessing_num_workers 32 \
    --log_max_eval_predictions 100 \
    --log_eval_predictions_fn "utils.log_predictions.write_predictions" \
    --log_examples 100 \
    --streaming True \
    --use_auth_token True \
    --dtype bfloat16 \
    --hub_private_repo True \
    --resume_from_checkpoint True \
    --ignore_data_skip \
    --gradient_checkpointing True \
    --pad_target_to_multiple_of 448 \
    --push_to_hub
